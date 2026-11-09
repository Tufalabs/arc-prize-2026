import copy
import unittest
import torch
from torch import nn
from training.flash.expert_lora import install_expert_lora, lora_delta, normalize_lora
from training.flash.pointwise_optimizations import expert_forward_with_weights
from sft.distillation.accumulation import validate_resume_manifest


def reference_grouped(x, w, offsets):
    starts = [0] + offsets.tolist()
    return torch.cat(
        [
            torch.nn.functional.linear(x[starts[i] : starts[i + 1]], w[i])
            for i in range(len(w))
        ]
    )


def module():
    m = nn.Module()
    m.gate_up_proj = nn.Parameter(torch.randn(3, 12, 8), requires_grad=False)
    m.down_proj = nn.Parameter(torch.randn(3, 8, 6), requires_grad=False)
    return m


class ExpertLoRATests(unittest.TestCase):
    def test_zero_init_and_dense_reference_all_parameter_gradients(self):
        torch.manual_seed(19)
        m = module()
        params = install_expert_lora(m, dict(rank=3, alpha=6, dropout=0))
        offsets = torch.tensor([2, 2, 5], dtype=torch.int32)
        x = torch.randn(5, 8, requires_grad=True)
        self.assertEqual(len(params), 6)
        self.assertTrue(
            torch.equal(
                lora_delta(m, "gate", x, offsets, reference_grouped), torch.zeros(5, 6)
            )
        )
        with torch.no_grad():
            m.lora_gate_B.normal_()
        actual = lora_delta(m, "gate", x, offsets, reference_grouped)
        merged = torch.bmm(m.lora_gate_B, m.lora_gate_A) * 2
        expected = reference_grouped(x, merged, offsets)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        upstream = torch.randn_like(actual)
        targets = (x, m.lora_gate_A, m.lora_gate_B)
        ga = torch.autograd.grad(actual, targets, upstream, retain_graph=True)
        ge = torch.autograd.grad(expected, targets, upstream)
        for a, b in zip(ga, ge):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        self.assertEqual(ga[1][1].count_nonzero(), 0)  # unused expert
        self.assertEqual(ga[2][1].count_nonzero(), 0)

    def test_optimizer_resume_and_frozen_base(self):
        torch.manual_seed(9)
        m = module()
        p = install_expert_lora(m, dict(rank=3, alpha=6, dropout=0))
        base = m.gate_up_proj.detach().clone()
        opt = torch.optim.AdamW(p.values(), lr=0.001, weight_decay=0)
        offsets = torch.tensor([1, 2, 3], dtype=torch.int32)

        def step(m, opt):
            opt.zero_grad()
            x = torch.ones(3, 8)
            out = torch.cat(
                [
                    lora_delta(m, n, x, offsets, reference_grouped)
                    for n in ["gate", "up"]
                ],
                -1,
            ) + reference_grouped(x, m.gate_up_proj, offsets)
            a, b = out.chunk(2, -1)
            hidden = torch.nn.functional.silu(a) * b
            y = reference_grouped(hidden, m.down_proj, offsets) + lora_delta(
                m, "down", hidden, offsets, reference_grouped
            )
            y.square().mean().backward()
            opt.step()

        step(m, opt)
        self.assertGreater(m.lora_down_B.count_nonzero(), 0)
        saved = copy.deepcopy(opt.state_dict())
        clone = copy.deepcopy(m)
        op2 = torch.optim.AdamW(
            [p for p in clone.parameters() if p.requires_grad], lr=0.001, weight_decay=0
        )
        op2.load_state_dict(saved)
        step(m, opt)
        step(clone, op2)
        for a, b in zip(m.parameters(), clone.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(base, m.gate_up_proj))
        self.assertIsNone(m.gate_up_proj.grad)
        self.assertGreater(m.lora_down_A.grad.count_nonzero(), 0)

    def test_resume_rejects_adapter_changes(self):
        a = {"expert_lora": dict(rank=8, alpha=16, dropout=0)}
        validate_resume_manifest(a, a)
        for b in [{}, {"expert_lora": dict(rank=16, alpha=16, dropout=0)}]:
            with self.assertRaises(ValueError):
                validate_resume_manifest(a, b)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU allocation qualification")
    def test_cuda_grouped_lora_forward_backward(self):
        from transformers.integrations.moe import _grouped_linear

        torch.manual_seed(17)
        m = nn.Module()
        m.gate_up_proj = nn.Parameter(
            torch.randn(4, 64, 32, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        m.down_proj = nn.Parameter(
            torch.randn(4, 32, 32, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        install_expert_lora(m, dict(rank=8, alpha=16, dropout=0))
        offsets = torch.tensor([13, 13, 29, 37], device="cuda", dtype=torch.int32)
        x = torch.randn(37, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad():
            m.lora_gate_B.normal_(std=0.02)
        actual = lora_delta(m, "gate", x, offsets, _grouped_linear)
        expected = lora_delta(m, "gate", x, offsets, reference_grouped)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.05)
        grad = torch.randn_like(actual)
        params = (x, m.lora_gate_A, m.lora_gate_B)
        a = torch.autograd.grad(actual, params, grad, retain_graph=True)
        b = torch.autograd.grad(expected, params, grad)
        for left, right in zip(a, b):
            self.assertTrue(torch.isfinite(left).all())
            torch.testing.assert_close(left, right, atol=0.02, rtol=0.08)
        self.assertEqual(a[1][1].count_nonzero(), 0)
        self.assertEqual(a[2][1].count_nonzero(), 0)

    def test_invalid_configuration(self):
        for value in [
            dict(rank=0, alpha=16, dropout=0),
            dict(rank=8, alpha=16, dropout=0.1),
            dict(rank=8, alpha=float("nan"), dropout=0),
        ]:
            with self.assertRaises(ValueError):
                normalize_lora(value)


def weighted_forward(m, x, ids, w):
    return expert_forward_with_weights(
        m, x, ids, w, m.gate_up_proj, m.down_proj, optimization="weighted_permutation"
    )


class IntegratedExpertLoRATests(unittest.TestCase):
    def test_expert_forward_zero_init_and_nonzero_reference(self):
        from transformers.integrations.moe import grouped_mm_experts_forward

        torch.manual_seed(7)
        m = nn.Module()
        m.gate_up_proj = nn.Parameter(torch.randn(3, 32, 16), requires_grad=False)
        m.down_proj = nn.Parameter(torch.randn(3, 16, 16), requires_grad=False)
        m.num_experts = 3
        m.has_gate = True
        m.has_bias = False
        m.is_transposed = False
        m._apply_gate = lambda z: (
            torch.nn.functional.silu(z.chunk(2, -1)[0]) * z.chunk(2, -1)[1]
        )
        x = torch.randn(7, 16, requires_grad=True)
        ids = torch.tensor([[0, 2]] * 7)
        w = torch.rand(7, 2, requires_grad=True)
        original = grouped_mm_experts_forward(m, x, ids, w)
        install_expert_lora(m, dict(rank=3, alpha=6, dropout=0))
        torch.testing.assert_close(weighted_forward(m, x, ids, w), original, atol=0, rtol=0)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.endswith("_B"):
                    p.normal_(std=0.01)
        actual = weighted_forward(m, x, ids, w)
        # Independent per-token/per-expert forward, including all three updates.
        vals = []
        for row in range(7):
            terms = []
            for slot in range(2):
                e = ids[row, slot]
                gu = m.gate_up_proj[e]
                gate = gu[:16] + 2 * m.lora_gate_B[e] @ m.lora_gate_A[e]
                up = gu[16:] + 2 * m.lora_up_B[e] @ m.lora_up_A[e]
                down = m.down_proj[e] + 2 * m.lora_down_B[e] @ m.lora_down_A[e]
                h = torch.nn.functional.silu(x[row] @ gate.T) * (x[row] @ up.T)
                terms.append((h @ down.T) * w[row, slot])
            vals.append(sum(terms))
        expected = torch.stack(vals)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        params = [x, w] + [p for p in m.parameters() if p.requires_grad]
        g = torch.randn_like(actual)
        a = torch.autograd.grad(actual, params, g, retain_graph=True)
        b = torch.autograd.grad(expected, params, g)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)


class PortIntegrationTests(unittest.TestCase):
    def test_install_preserves_rng_and_rejects_duplicate_install(self):
        m = module()
        before = torch.get_rng_state().clone()
        install_expert_lora(m, dict(rank=3, alpha=6, dropout=0))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_expert_lora(m, dict(rank=3, alpha=6, dropout=0))

    def test_policy_and_config_roundtrip_expert_only(self):
        import json
        import tempfile
        from pathlib import Path
        from training.flash.policies import TrainabilityPolicy
        from sft.distillation.config import load_config
        from sft.distillation.train import worker_command
        from tests.distillation.test_entrypoint import config

        lora = dict(rank=8, alpha=16, dropout=0)
        policy = TrainabilityPolicy(mode="experts", expert_lora=lora)
        self.assertEqual(policy.expert_lora, lora)
        with self.assertRaises(ValueError):
            TrainabilityPolicy(mode="experts")
        with self.assertRaises(ValueError):
            TrainabilityPolicy(mode="frozen", expert_lora=lora)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            c = config() | dict(expert_lora=lora, train_routers=False)
            path.write_text(json.dumps(c))
            loaded = load_config(path)
            command = worker_command(loaded, Path(tmp))
            self.assertIn("--no-train-routers", command)
            self.assertEqual(
                json.loads(command[command.index("--expert-lora") + 1]), lora
            )
            for override in (
                dict(expert_lora=None),
                dict(expert_optimization="none"),
                dict(train_routers=1),
            ):
                path.write_text(json.dumps(c | override))
                with self.assertRaises(ValueError):
                    load_config(path)
        validate_resume_manifest({}, {"expert_lora": None, "train_routers": True})
        for override in ({"train_routers": False}, {"expert_lora": lora}):
            with self.assertRaises(ValueError):
                validate_resume_manifest({}, override)

    def test_recipe_returns_all_trainables_in_legacy_layer_order(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from training.flash.model import FlashModelBundle
        from sft.distillation.flash_model import build_model

        lora = dict(rank=8, alpha=16, dropout=0)
        for train_routers in (True, False):
            model = nn.Module()
            model.config = SimpleNamespace(
                num_hidden_layers=2, num_experts=3, hidden_size=4
            )
            groups = {"routers": {}, "expert_lora": {}}
            for layer in range(2):
                if train_routers:
                    p = nn.Parameter(torch.ones(3, 4))
                    model.register_parameter(f"router{layer}", p)
                    groups["routers"][f"layers.{layer}.mlp.gate.weight"] = p
                for proj in ("gate", "up", "down"):
                    for side in ("A", "B"):
                        p = nn.Parameter(torch.ones(1))
                        model.register_parameter(f"{layer}_{proj}_{side}", p)
                        groups["expert_lora"][
                            f"layers.{layer}.mlp.experts.lora_{proj}_{side}"
                        ] = p
            bundle = FlashModelBundle(model, nn.Identity(), groups)
            with patch(
                "sft.distillation.flash_model.build_flash_model", return_value=bundle
            ) as builder:
                _, _, parameters = build_model(
                    "/unused", expert_lora=lora, train_routers=train_routers
                )
            self.assertEqual(len(parameters), 14 if train_routers else 12)
            self.assertEqual(
                list(parameters), sorted(parameters, key=lambda x: int(x.split(".")[1]))
            )
            self.assertEqual(
                builder.call_args.kwargs["trainability"].mode,
                "routers" if train_routers else "experts",
            )


class ExportTests(unittest.TestCase):
    def test_export_merges_all_experts_and_refuses_incomplete_checkpoint(self):
        import json
        import tempfile
        from pathlib import Path
        from safetensors.torch import save_file, load_file
        from sft.distillation.common import sha
        from sft.distillation.export import export_checkpoint, PREFIX
        from training.flash.reader import dequant_nvfp4
        from training.flash.expert_lora import merge_projection

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            config = {
                "text_config": {"num_hidden_layers": 1, "num_experts": 2},
                "quantization_config": {"quant_algo": "NVFP4"},
            }
            (base / "config.json").write_text(json.dumps(config))
            (base / "distillation-parent.json").write_text("{}")
            tensors, trainables = {}, {}
            for projection in ("gate", "up", "down"):
                name = f"layers.0.mlp.experts.lora_{projection}"
                trainables[name + "_A"] = torch.randn(2, 3, 16) * 0.1
                trainables[name + "_B"] = torch.randn(2, 16, 3) * 0.1
                for expert in range(2):
                    key = PREFIX + f"layers.0.mlp.experts.{expert}.{projection}_proj"
                    tensors[key + ".weight"] = torch.full(
                        (16, 8), 0x21, dtype=torch.uint8
                    )
                    tensors[key + ".weight_scale"] = torch.full((16, 1), 0.125)
                    tensors[key + ".weight_scale_2"] = torch.tensor(1.0)
                    tensors[key + ".input_scale"] = torch.tensor(1.0)
            router = "layers.0.mlp.gate.weight"
            tensors[PREFIX + router] = torch.zeros(2, 16)
            trainables[router] = torch.randn(2, 16)
            save_file(tensors, base / "model.safetensors")
            (base / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
            )
            lora = dict(rank=3, alpha=6, dropout=0)
            manifest = {
                "expert_lora": lora,
                **{
                    field: sha(base / file)
                    for field, file in [
                        ("student_config_sha256", "config.json"),
                        ("student_index_sha256", "model.safetensors.index.json"),
                        ("pruning_sha256", "distillation-parent.json"),
                    ]
                },
            }
            state = {
                "manifest": manifest,
                "routers": trainables,
                "router_names": list(trainables),
                "step": 2,
            }
            checkpoint = root / "checkpoint.pt"
            torch.save(state, checkpoint)
            out = root / "out"
            report = export_checkpoint(base, checkpoint, out)
            self.assertEqual(report["merged_expert_weights"], 6)
            values = load_file(out / "model.safetensors")
            self.assertFalse(any("scale" in key for key in values))
            self.assertNotIn(
                "quantization_config", json.loads((out / "config.json").read_text())
            )
            for projection in ("gate", "up", "down"):
                for expert in range(2):
                    key = PREFIX + f"layers.0.mlp.experts.{expert}.{projection}_proj"
                    weight = dequant_nvfp4(
                        tensors[key + ".weight"],
                        tensors[key + ".weight_scale"],
                        tensors[key + ".weight_scale_2"],
                    )
                    name = f"layers.0.mlp.experts.lora_{projection}"
                    expected = merge_projection(
                        weight,
                        trainables[name + "_A"][expert],
                        trainables[name + "_B"][expert],
                        lora,
                    )
                    torch.testing.assert_close(
                        values[key + ".weight"], expected, rtol=0, atol=0
                    )
            torch.testing.assert_close(values[PREFIX + router], trainables[router])
            with self.assertRaises(FileExistsError):
                export_checkpoint(base, checkpoint, out)
            # Expert-only export preserves the frozen source router.
            state["manifest"]["train_routers"] = False
            state["routers"].pop(router)
            state["router_names"] = list(state["routers"])
            torch.save(state, checkpoint)
            export_checkpoint(base, checkpoint, root / "expert-only")
            torch.testing.assert_close(
                load_file(root / "expert-only/model.safetensors")[PREFIX + router],
                tensors[PREFIX + router],
            )
            state["routers"].pop(next(iter(state["routers"])))
            state["router_names"] = list(state["routers"])
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "exactly"):
                export_checkpoint(base, checkpoint, root / "invalid")
            self.assertFalse((root / "invalid").exists())
