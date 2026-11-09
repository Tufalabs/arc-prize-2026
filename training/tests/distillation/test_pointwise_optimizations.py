import types
import unittest

import torch

from sft.distillation.accumulation import validate_resume_manifest
from training.flash.norm_tail import grouped_norm_forward
from training.flash.pointwise_optimizations import expert_forward_with_weights


class PointwiseTests(unittest.TestCase):
    def test_expert_modes_match_reference_outputs_and_gradients(self):
        from transformers.integrations.moe import grouped_mm_experts_forward

        def gate(x):
            a, b = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(a) * b

        torch.manual_seed(7)
        m = types.SimpleNamespace(
            num_experts=4,
            has_gate=True,
            has_bias=False,
            is_transposed=False,
            gate_up_proj=torch.randn(4, 32, 16),
            down_proj=torch.randn(4, 16, 16),
            _apply_gate=gate,
        )
        for n in (1, 7, 32):
            ids = torch.stack([torch.randperm(4)[:2] for _ in range(n)])
            x = torch.randn(n, 16, requires_grad=True)
            w = torch.randn(n, 2, requires_grad=True)
            expected = grouped_mm_experts_forward(m, x, ids, w)
            upstream = torch.randn_like(expected)
            grads = torch.autograd.grad(expected, (x, w), upstream, retain_graph=True)
            for mode in ("no_sentinel", "weighted_permutation"):
                actual = expert_forward_with_weights(
                    m, x, ids, w, m.gate_up_proj, m.down_proj, optimization=mode
                )
                actual_grads = torch.autograd.grad(actual, (x, w), upstream)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(
                    all(torch.equal(a, b) for a, b in zip(actual_grads, grads))
                )

    def test_norm_tail_preserves_fp32_bf16_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(9)
            x = torch.randn(2, 7, 256, dtype=dtype, requires_grad=True)
            m = types.SimpleNamespace(
                group_size=64, eps=1e-6, weight=torch.randn(256, dtype=dtype)
            )
            xf = x.float().reshape(2, 7, 4, 64)
            ref = (
                (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + m.eps)).flatten(-2)
                * (1 + m.weight.float())
            ).to(dtype)
            actual = grouped_norm_forward(m, x)
            upstream = torch.randn_like(ref)
            self.assertTrue(torch.equal(actual, ref))
            self.assertTrue(
                torch.equal(
                    torch.autograd.grad(actual, x, upstream)[0],
                    torch.autograd.grad(ref, x, upstream)[0],
                )
            )

    def test_expert_forward_requires_an_explicit_optimized_path(self):
        m = types.SimpleNamespace(num_experts=2, _apply_gate=lambda z: z)
        x, ids, w = torch.zeros(1, 4), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)
        for mode in ("none", "typo", None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "optimization="):
                expert_forward_with_weights(
                    m, x, ids, w, torch.zeros(2, 8, 4), torch.zeros(2, 4, 4), optimization=mode
                )

    def test_resume_metadata_defaults_and_changes(self):
        validate_resume_manifest(
            {}, {"expert_optimization": "none", "fuse_norm_tail": False}
        )
        with self.assertRaises(ValueError):
            validate_resume_manifest({}, {"expert_optimization": "no_sentinel"})
        with self.assertRaises(ValueError):
            validate_resume_manifest({}, {"fuse_norm_tail": True})

    def test_checkpoints_saved_with_retired_placement_resume(self):
        # Manifests recorded the placement until NVFP4 expert storage was
        # removed; a checkpoint saved with it resumes with BF16 experts.
        for storage, distribution in [
            ("bf16", "replicated"), ("nvfp4", "replicated"), ("nvfp4", "fsdp")
        ]:
            saved = {
                "expert_storage": storage,
                "nvfp4_distribution": distribution,
                "model_distribution": "fsdp",
            }
            self.assertFalse(validate_resume_manifest(saved, {}))
            self.assertIn("expert_storage", saved)  # the saved manifest is untouched
        # The per-GPU replica existed only with NVFP4 storage; unknown values
        # were never valid.
        for saved in [
            {"model_distribution": "replicated"},
            {"expert_storage": "fp8"},
            {"nvfp4_distribution": "bad"},
        ]:
            with self.subTest(saved=saved), self.assertRaisesRegex(
                ValueError, "retired setting accepts only"
            ):
                validate_resume_manifest(saved, {})


class PointwiseIntegrationTests(unittest.TestCase):
    def test_new_config_defaults_and_explicit_opt_out_reach_worker(self):
        import json
        import tempfile
        from pathlib import Path

        from sft.distillation.config import load_config
        from sft.distillation.train import worker_command

        base = dict(
            teacher="/t",
            student="/s",
            data="/d",
            data_sha256="sha",
            cache="/c",
            probe={"cache": "/v"},
            wandb={"mode": "offline"},
            steps=2,
            save_every=1,
            chunk=64,
            seed=1,
            lr=5e-6,
            temperature=1.0,
            ce_weight=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for overrides, expert, norm in [
                ({}, "weighted_permutation", True),
                (
                    {"expert_optimization": "none", "fuse_norm_tail": False},
                    "none",
                    False,
                ),
            ]:
                cfg = dict(base, **overrides)
                path.write_text(json.dumps(cfg))
                loaded = load_config(path)
                self.assertEqual(loaded["expert_optimization"], expert)
                self.assertEqual(loaded["fuse_norm_tail"], norm)
                self.assertEqual(loaded["loss_weighting"], "example")
                for values in [cfg, loaded]:
                    cmd = worker_command(values, Path("/result"))
                    self.assertEqual(
                        cmd[cmd.index("--expert-optimization") + 1], expert
                    )
                    self.assertIn(
                        "--fuse-norm-tail" if norm else "--no-fuse-norm-tail", cmd
                    )
                if norm:
                    for allow in [False, True]:
                        with self.assertRaises(ValueError):
                            validate_resume_manifest(
                                {},
                                {"expert_optimization": expert, "fuse_norm_tail": norm},
                                allow,
                            )
            for overrides in [
                {"expert_optimization": "bad"},
                {"expert_storage": "bad"},
                {"nvfp4_distribution": "bad"},
                {"model_distribution": "bad"},
                {"fuse_norm_tail": 1},
            ]:
                path.write_text(json.dumps(dict(base, **overrides)))
                with self.assertRaises(ValueError):
                    load_config(path)

            # Configs and attempt snapshots written while expert storage was
            # a choice load as the BF16 FSDP layout.
            path.write_text(json.dumps(dict(base)))
            expected = load_config(path)
            path.write_text(
                json.dumps(
                    dict(
                        base,
                        expert_storage="nvfp4",
                        nvfp4_distribution="fsdp",
                        model_distribution="fsdp",
                        world_size=2,
                    )
                )
            )
            loaded = load_config(path)
            self.assertEqual(loaded, dict(expected, world_size=2))
            cmd = worker_command(loaded, Path("/result"))
            self.assertIn("--nproc-per-node=2", cmd)
            self.assertFalse(
                {"--expert-storage", "--nvfp4-distribution", "--model-distribution"}
                & set(cmd)
            )

            for overrides in [
                {"world_size": 1},
                {"model_distribution": "replicated", "world_size": 1},
                {"model_distribution": "replicated"},
                {"world_size": 0},
            ]:
                path.write_text(json.dumps(dict(base, **overrides)))
                with self.assertRaises(ValueError):
                    load_config(path)
