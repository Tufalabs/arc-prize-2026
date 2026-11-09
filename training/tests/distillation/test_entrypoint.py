import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft.distillation.config import load_config, validate_inputs
from sft.distillation.launch import endpoint
from sft.distillation.launch import main as launch
from sft.distillation.train import main, worker_command


def config():
    return {
        "teacher": "/teacher",
        "student": "/student",
        "data": "/data",
        "data_sha256": "sha",
        "cache": "/cache",
        "probe": {"cache": "/probe-cache"},
        "wandb": {"mode": "offline"},
        "steps": 1481,
        "production_total_steps": 1481,
        "save_every": 25,
        "seed": 20260916,
        "chunk": 64,
        "temperature": 1.0,
        "ce_weight": 0.0,
        "lr": 1e-5,
    }


class EntrypointTests(unittest.TestCase):
    def test_qsa_default_reaches_config_and_worker(self):
        for backend, expected in (("flex_qsa", "physical_sparse"), ("sdpa", "flex")):
            with self.subTest(backend=backend):
                c = config()
                c["attention_backend"] = backend
                command = worker_command(c, Path("/result"))
                self.assertEqual(command[command.index("--qsa-kernel") + 1], expected)
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / "config.json"
                    p.write_text(json.dumps(c))
                    self.assertEqual(load_config(p)["qsa_kernel"], expected)

    def test_sparse_kernel_config_and_command(self):
        for kernel in ("flex", "indexed_backward", "physical_sparse", "group_sparse"):
            c = config()
            c.update(attention_backend="flex_qsa", qsa_kernel=kernel)
            command = worker_command(c, Path("/result"))
            self.assertEqual(command[command.index("--qsa-kernel") + 1], kernel)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "config.json"
                p.write_text(json.dumps(c))
                self.assertEqual(load_config(p)["qsa_kernel"], kernel)
                if kernel != "flex":
                    c["attention_backend"] = "sdpa"
                    p.write_text(json.dumps(c))
                    with self.assertRaisesRegex(ValueError, "requires attention_backend"):
                        load_config(p)
                    c["attention_backend"] = "flex_qsa"
                    c["sequence_packing"] = {"enabled": True, "version": 1}
                    p.write_text(json.dumps(c))
                    with self.assertRaisesRegex(ValueError, "sequence_packing is no longer supported"):
                        load_config(p)

    def test_default_attention_retention_and_opt_out(self):
        from training.flash.offload import resolve_qsa_retention

        self.assertTrue(resolve_qsa_retention("flex_qsa", True))
        self.assertFalse(resolve_qsa_retention("sdpa", False))
        self.assertFalse(resolve_qsa_retention("flex_qsa", False))
        for explicit in (None, False, True):
            c = config()
            c.update(attention_backend="flex_qsa", reuse_qsa_selection=True)
            if explicit is not None:
                c["retain_qsa_attention"] = explicit
            command = worker_command(c, Path("/result"))
            expected = explicit is not False
            flag = "--retain-qsa-attention" if expected else "--no-retain-qsa-attention"
            self.assertIn(flag, command)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "config.json"
                p.write_text(json.dumps(c))
                self.assertEqual(load_config(p)["retain_qsa_attention"], expected)

    def test_default_worker_batching_and_explicit_random(self):
        from sft.distillation.batching import normalize_batching

        c = config()
        for explicit in [None, {"policy": "random"}]:
            if explicit is not None:
                c["batching"] = explicit
            command = worker_command(c, Path("/result"))
            actual = normalize_batching(
                json.loads(command[command.index("--batching-config") + 1])
            )
            self.assertEqual(
                actual["policy"], "length_bucket" if explicit is None else "random"
            )
            if explicit is None:
                self.assertEqual(actual["pool_size"], 128)

    def test_attention_retention_config_and_resume(self):
        from sft.distillation.accumulation import validate_resume_manifest

        c = config()
        self.assertNotIn("--retain-qsa-attention", worker_command(c, Path("/result")))
        c.update(
            attention_backend="flex_qsa",
            reuse_qsa_selection=True,
            retain_qsa_attention=True,
        )
        self.assertIn("--retain-qsa-attention", worker_command(c, Path("/result")))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps(c))
            self.assertTrue(load_config(p)["retain_qsa_attention"])
            for override in (
                {"retain_qsa_attention": 1},
                {"reuse_qsa_selection": False},
                {"attention_backend": "sdpa"},
            ):
                p.write_text(json.dumps(c | override))
                with self.assertRaises(ValueError):
                    load_config(p)
        validate_resume_manifest({}, {"retain_qsa_attention": False})
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest({}, {"retain_qsa_attention": True})

    def test_batching_config_roundtrip_and_accumulation_requirement(self):
        c = config()
        c["batching"] = {"policy": "length_bucket", "pool_size": 256}
        command = worker_command(c, Path("/result"))
        self.assertEqual(
            json.loads(command[command.index("--batching-config") + 1]), c["batching"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps(c))
            self.assertEqual(load_config(p)["batching"], c["batching"])
            c["batching"] = {"policy": "length_within_update"}
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, "requires gradient_accumulation"):
                load_config(p)
            c["gradient_accumulation"] = 4
            p.write_text(json.dumps(c))
            self.assertEqual(load_config(p)["batching"], c["batching"])

    def test_sequence_packing_is_rejected(self):
        c = config()
        c["sequence_packing"] = {"enabled": True}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                load_config(p)

    def test_validate_only_never_launches_or_writes_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "config.json"
            p.write_text(json.dumps(config()))
            with (
                patch(
                    "sft.distillation.train.validate_inputs",
                    return_value={"rows": 5924},
                ),
                patch("subprocess.Popen") as process,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                main(
                    [
                        "--config",
                        str(p),
                        "--result",
                        str(root / "result"),
                        "--validate-only",
                    ]
                )
            process.assert_not_called()
            self.assertFalse((root / "result").exists())
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "validated")
            self.assertIn("sft.distillation.worker", report["command"])
            self.assertNotIn("sft.distillation.teacher_cache", report["command"])

    def test_invalid_objective_and_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            for key, value in [
                ("temperature", 0),
                ("lr", float("nan")),
                ("teacher_mode", "online"),
                ("world_size", 8),
                ("world_size", 0),
                ("world_size", True),
                ("model_distribution", "replicated"),
                ("steps", 1482),
                ("gradient_accumulation", 0),
                ("gradient_accumulation", True),
                ("data_microbatches", -1),
            ]:
                c = config()
                c[key] = value
                p.write_text(json.dumps(c))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_config(p)

    def test_missing_teacher_cache_fails_before_probe_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = config()
            c["cache"] = str(Path(tmp) / "missing")
            with (
                patch("sft.distillation.common.validate_roles", return_value={}),
                patch("sft.distillation.common.sha", return_value="sha"),
                patch("sft.distillation.pruning_probe.validate_probe") as probe,
            ):
                with self.assertRaisesRegex(
                    ValueError, "Completed teacher cache required"
                ):
                    validate_inputs(c)
                probe.assert_not_called()

    def test_worker_preserves_resume_and_total_step_axis(self):
        c = config()
        c["resume"] = "/checkpoint-step100.pt"
        c["steps"] = 200
        command = worker_command(c, Path("/result"))
        self.assertEqual(command[command.index("--steps") + 1], "200")
        self.assertEqual(command[command.index("--resume") + 1], c["resume"])
        self.assertIn("--module", command)

    def test_ga2_worker_arguments_and_external_resume_budget(self):
        import torch

        c = config()
        c.update(
            gradient_accumulation=2,
            data_microbatches=1481,
            allow_accumulation_change=True,
            steps=766,
            production_total_steps=766,
        )
        command = worker_command(c, Path("/result"))
        self.assertEqual(command[command.index("--gradient-accumulation") + 1], "2")
        self.assertEqual(command[command.index("--data-microbatches") + 1], "1481")
        self.assertIn("--allow-accumulation-change", command)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp = root / "checkpoint.pt"
            torch.save(
                {
                    "step": 50,
                    "manifest": {},
                    "rng": [{}] * 4,
                    "frozen": [{}] * 4,
                    "optimizer": {"state": {0: {}}},
                    "routers": {"r": torch.ones(2)},
                },
                cp,
            )
            base = root / "base.json"
            base.write_text(json.dumps(c))
            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc"],
                ),
                patch("sft.distillation.launch.subprocess.run") as submit,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                launch(
                    [
                        "--config",
                        str(base),
                        "--run-dir",
                        str(root / "run"),
                        "--resume-checkpoint",
                        str(cp),
                        "--segment-steps",
                        "1000",
                    ]
                )
            submit.assert_not_called()
            report = json.loads(output.getvalue())
            self.assertEqual((report["start_step"], report["end_step"]), (50, 766))
            self.assertEqual(report["initial_resume"], str(cp.resolve()))
            self.assertEqual(len(report["resume_sha256"]), 64)

    def test_segments_cover_frozen_dataset_exactly(self):
        ends = []
        n = 0
        while n < 1481:
            n = endpoint(n, 1481)
            ends.append(n)
        self.assertEqual(ends[-2:], [1400, 1481])
        self.assertEqual(len(ends), 15)
        with self.assertRaises(ValueError):
            endpoint(1481, 1481)

    def test_launch_dry_run_pins_source_and_does_not_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "base.json"
            p.write_text(json.dumps(config()))
            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc123"],
                ) as check,
                patch("sft.distillation.launch.subprocess.run") as run,
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                launch(["--config", str(p), "--run-dir", str(root / "run")])
            run.assert_not_called()
            self.assertEqual(check.call_count, 2)
            self.assertFalse((root / "run/attempts.json").exists())
            report = json.loads(out.getvalue())
            self.assertEqual(report["end_step"], 100)
            self.assertEqual(report["git_commit"], "abc123")
            self.assertIn("abc123", report["command"])

    def test_dirty_worktree_cannot_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "base.json"
            p.write_text(json.dumps(config()))
            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    return_value=" M file.py",
                ),
                patch("sft.distillation.launch.subprocess.run") as run,
                self.assertRaisesRegex(ValueError, "Commit the worktree"),
            ):
                launch(["--config", str(p), "--run-dir", str(root / "run"), "--submit"])
            run.assert_not_called()

    def test_submit_pins_exact_config_and_records_job(self):
        from sft.distillation.cache_transaction import digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "base.json"
            p.write_text(json.dumps(config()))
            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc123", "12345\n"],
                ) as check,
                patch("sft.distillation.launch.subprocess.run") as admission,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch(["--config", str(p), "--run-dir", str(root / "run"), "--submit"])
            saved = root / "run/attempt001/config.json"
            row = json.loads((root / "run/attempts.json").read_text())[0]
            self.assertEqual(row["status"], "submitted")
            self.assertEqual(row["job_id"], "12345")
            self.assertEqual(row["config_sha256"], digest(saved))
            self.assertEqual(check.call_args.args[0][-1], digest(saved))
            self.assertIn("--test-only", admission.call_args.args[0])
            self.assertIn("--gpus=b200:4", admission.call_args.args[0])
            self.assertIn("--account=research", admission.call_args.args[0])
            self.assertIn("--partition=dev", admission.call_args.args[0])
            self.assertIn("--qos=b200-pool", admission.call_args.args[0])
            self.assertEqual(json.loads(saved.read_text())["steps"], 100)

    def test_rejected_admission_leaves_no_attempt_and_retry_can_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "base.json"
            p.write_text(json.dumps(config()))
            run_dir = root / "run"

            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc123"],
                ),
                patch(
                    "sft.distillation.launch.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["sbatch"]),
                ),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                launch(["--config", str(p), "--run-dir", str(run_dir), "--submit"])

            self.assertFalse((run_dir / "attempt001").exists())
            self.assertFalse((run_dir / "attempts.json").exists())

            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc123", "12345\n"],
                ),
                patch("sft.distillation.launch.subprocess.run") as admission,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch(["--config", str(p), "--run-dir", str(run_dir), "--submit"])

            self.assertIn("--test-only", admission.call_args.args[0])
            self.assertTrue((run_dir / "attempt001/config.json").is_file())
            row = json.loads((run_dir / "attempts.json").read_text())[0]
            self.assertEqual((row["status"], row["job_id"]), ("submitted", "12345"))

    def test_uncertain_submission_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "base.json"
            p.write_text(json.dumps(config()))
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "attempts.json").write_text(
                json.dumps([{"status": "submission_intent"}])
            )
            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc123"],
                ) as check,
                self.assertRaisesRegex(ValueError, "Uncertain previous submission"),
            ):
                launch(["--config", str(p), "--run-dir", str(run_dir), "--submit"])
            self.assertEqual(check.call_count, 2)

    def test_real_sbatch_rejection_is_recorded_and_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "base.json"
            p.write_text(json.dumps(config()))
            run_dir = root / "run"
            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=[
                        "",
                        "abc123",
                        subprocess.CalledProcessError(1, ["sbatch"]),
                    ],
                ),
                patch("sft.distillation.launch.subprocess.run"),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                launch(["--config", str(p), "--run-dir", str(run_dir), "--submit"])
            rejected = json.loads((run_dir / "attempts.json").read_text())[0]
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(rejected["returncode"], 1)
            self.assertFalse((run_dir / "attempt001").exists())

            with (
                patch("sft.distillation.launch.validate_inputs"),
                patch(
                    "sft.distillation.launch.subprocess.check_output",
                    side_effect=["", "abc123", "12345\n"],
                ),
                patch("sft.distillation.launch.subprocess.run"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch(["--config", str(p), "--run-dir", str(run_dir), "--submit"])
            attempts = json.loads((run_dir / "attempts.json").read_text())
            self.assertEqual(
                [row["status"] for row in attempts], ["rejected", "submitted"]
            )
            self.assertTrue((run_dir / "attempt002/config.json").is_file())
