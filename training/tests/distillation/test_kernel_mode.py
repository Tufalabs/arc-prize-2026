"""kernel_mode in the distillation config, supervisor command, worker and resume manifest."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from sft.distillation.accumulation import validate_resume_manifest
from sft.distillation.config import WORKER_KERNEL_FLAGS, kernel_stack, load_config, worker_kernel_stack
from sft.distillation.train import worker_command
from training.flash.kernel_mode import KERNEL_FIELDS, PRESETS, resolve_kernel_stack


def config(**extra):
    return {
        "teacher": "/teacher",
        "student": "/student",
        "data": "/data",
        "data_sha256": "sha",
        "cache": "/cache",
        "probe": {"cache": "/probe-cache"},
        "wandb": {"mode": "offline"},
        "steps": 10,
        "save_every": 5,
        "seed": 20260916,
        "chunk": 64,
        "temperature": 1.0,
        "ce_weight": 0.0,
        "lr": 5e-6,
        **extra,
    }


# The settings of the published router-only e205/e257/e276 configs.
PUBLISHED = dict(
    attention_backend="flex_qsa",
    reuse_qsa_selection=True,
    retain_qsa_attention=True,
    offload_every=3,
    gradient_accumulation=8,
)


def load(payload):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(payload))
        return load_config(path)


def worker_args(command):
    """The worker's kernel flags, parsed back from the supervisor's command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-mode")
    parser.add_argument("--kernel-options")
    for name in WORKER_KERNEL_FLAGS:
        if name != "fuse_norm_tail":
            parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--fuse-norm-tail", action=argparse.BooleanOptionalAction)
    args, _ = parser.parse_known_args(command)
    return args


class KernelModeConfigTests(unittest.TestCase):
    def test_default_is_the_reference_stack(self):
        loaded = load(config())
        self.assertEqual(loaded["kernel_mode"], "reference")
        self.assertEqual(
            {name: loaded[name] for name in KERNEL_FIELDS}, resolve_kernel_stack("reference")
        )
        self.assertFalse(loaded["gdn_guard"])

    def test_published_configs_keep_their_stack(self):
        loaded = load(config(**PUBLISHED))
        self.assertEqual(
            (loaded["attention_backend"], loaded["qsa_kernel"], loaded["checkpoint_policy"]),
            ("flex_qsa", "physical_sparse", "full"),
        )
        self.assertEqual(
            (loaded["expert_optimization"], loaded["fuse_norm_tail"]),
            ("weighted_permutation", True),
        )
        self.assertTrue(loaded["reuse_qsa_selection"] and loaded["retain_qsa_attention"])

    def test_fast_switches_the_stack_and_keeps_placement(self):
        loaded = load(config(kernel_mode="fast", **PUBLISHED))
        self.assertEqual(
            {name: loaded[name] for name in KERNEL_FIELDS}, resolve_kernel_stack("fast")
        )
        self.assertEqual(loaded["qsa_kernel"], "group_sparse")
        self.assertTrue(loaded["gdn_guard"])
        # Offload, selection reuse and retention are the config's, not the switch's.
        self.assertEqual(loaded["offload_every"], 3)
        self.assertTrue(loaded["reuse_qsa_selection"] and loaded["retain_qsa_attention"])

    def test_explicit_fields_override_the_preset(self):
        loaded = load(config(kernel_mode="fast", selector_policy="batched_fp32_top513_v1",
                             checkpoint_policy="full"))
        expected = resolve_kernel_stack("fast") | {
            "selector_policy": "batched_fp32_top513_v1", "checkpoint_policy": "full"}
        self.assertEqual({name: loaded[name] for name in KERNEL_FIELDS}, expected)
        # A loaded snapshot resolves to itself (the attempt config the job runs).
        self.assertEqual(load(loaded)["kernel_mode"], "fast")
        self.assertEqual({name: load(loaded)[name] for name in KERNEL_FIELDS}, expected)

    def test_the_replicated_layout_is_retired(self):
        # It required NVFP4 expert storage; experts are now always BF16 and
        # FSDP-sharded, so fast replicates the frozen routers everywhere.
        self.assertTrue(load(config(kernel_mode="fast", model_distribution="fsdp"))[
            "replicate_frozen_routers"])
        with self.assertRaisesRegex(ValueError, "retired setting accepts only"):
            load(config(kernel_mode="fast", model_distribution="replicated", world_size=1,
                        expert_storage="nvfp4"))

    def test_retired_expert_storage_loads_with_one_warning(self):
        from unittest import mock

        import sft.distillation.config as config_module

        retired = dict(expert_storage="nvfp4", nvfp4_distribution="replicated")
        with mock.patch.object(config_module, "_warned_retired", set()), self.assertLogs(
            config_module.log, "WARNING"
        ) as logs:
            for _ in range(2):
                self.assertEqual(load(config(kernel_mode="fast", **retired)),
                                 load(config(kernel_mode="fast")))
        self.assertEqual(
            sorted(message.split("retired setting ")[1].split()[0] for message in logs.output),
            ["expert_storage", "nvfp4_distribution"],
        )

    def test_invalid_switch_and_fields(self):
        for extra in (
            {"kernel_mode": "turbo"},
            {"kernel_mode": "fast", "gdn_convolution": "cuda"},
            {"expert_optimization": "fused"},
            {"fuse_norm_tail": 1},
            {"attention_backend": "flash"},
            {"kernel_mode": "fast", "attention_backend": "sdpa", "qsa_kernel": "group_sparse"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                load(config(**extra))

    def test_worker_builds_exactly_the_supervisor_stack(self):
        for extra in ({}, PUBLISHED, {"kernel_mode": "fast", **PUBLISHED},
                      {"kernel_mode": "fast", "checkpoint_policy": "full", "gdn_guard": False}):
            with self.subTest(extra=extra):
                command = worker_command(config(**extra), Path("/result"))
                args = worker_args(command)
                stack = kernel_stack(config(**extra))
                self.assertEqual(args.kernel_mode, stack.pop("kernel_mode"))
                self.assertEqual(worker_kernel_stack(args), stack)

    def test_worker_without_flags_builds_the_reference_stack(self):
        args = argparse.Namespace(
            kernel_mode="reference", kernel_options="{}", **dict.fromkeys(WORKER_KERNEL_FLAGS),
        )
        self.assertEqual(worker_kernel_stack(args), resolve_kernel_stack("reference"))
        # A flag left unset never masks the same field passed in --kernel-options.
        args.kernel_options = json.dumps({"attention_backend": "flex_qsa"})
        self.assertEqual(worker_kernel_stack(args)["qsa_kernel"], "physical_sparse")


class KernelModeResumeTests(unittest.TestCase):
    def manifest(self, mode, **extra):
        stack = resolve_kernel_stack(mode, {"attention_backend": "flex_qsa"})
        selector = stack.pop("selector_policy")
        return {"kernel_mode": mode, **stack, "qsa_selector_policy": selector, "lr": 5e-6, **extra}

    def test_manifests_from_before_the_switch_resume_as_reference(self):
        legacy = {
            "attention_backend": "flex_qsa",
            "qsa_kernel": "physical_sparse",
            "expert_optimization": "weighted_permutation",
            "fuse_norm_tail": True,
            "expert_storage": "bf16",
            "qsa_selector_policy": "batched_fp32_top513_v1",
            "lr": 5e-6,
        }
        self.assertFalse(validate_resume_manifest(legacy, self.manifest("reference")))
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest(legacy, self.manifest("fast"))

    def test_a_resume_cannot_switch_stacks_or_fields(self):
        self.assertFalse(validate_resume_manifest(self.manifest("fast"), self.manifest("fast")))
        for current in (
            self.manifest("reference"),
            self.manifest("fast", gdn_guard=False),
            self.manifest("fast", checkpoint_policy="full"),
            self.manifest("fast", kernel_mode="reference"),
        ):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_resume_manifest(self.manifest("fast"), current)

    def test_preset_names_every_manifest_field(self):
        self.assertEqual(set(PRESETS["reference"]), set(PRESETS["fast"]))
        self.assertEqual(set(self.manifest("fast")) - {"kernel_mode", "qsa_selector_policy", "lr"},
                         set(KERNEL_FIELDS) - {"selector_policy"})


if __name__ == "__main__":
    unittest.main()
