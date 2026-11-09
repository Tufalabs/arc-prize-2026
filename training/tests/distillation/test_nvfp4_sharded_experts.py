"""Opt-in expert_storage="nvfp4_sharded": packed experts expanded per forward equal the default BF16 storage."""

import unittest

import torch

from training.flash.expert_lora import install_expert_lora
from training.flash.nvfp4_experts import NVFP4Experts
from training.flash.pointwise_optimizations import expert_forward_with_weights
from training.flash.policies import DistributionPolicy
from training.flash.reader import dequant_nvfp4


def reference_grouped(x, w, offsets):
    starts = [0] + offsets.tolist()
    return torch.cat([torch.nn.functional.linear(x[starts[i] : starts[i + 1]], w[i]) for i in range(len(w))])


def packed_projection(experts, rows, columns, generator):
    """Random NVFP4 bytes with positive FP8 block scales and per-expert tensor scales."""
    packed = torch.randint(0, 256, (experts, rows, columns // 2), dtype=torch.uint8, generator=generator)
    scale = (torch.rand(experts, rows, columns // 16, generator=generator) * 2 + 0.25).to(torch.float8_e4m3fn)
    outer = torch.rand(experts, generator=generator) * 0.01 + 0.001
    return packed, scale, outer


def tensors(experts=3, intermediate=16, hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, rows, columns in (("gate", intermediate, hidden), ("up", intermediate, hidden),
                                ("down", hidden, intermediate)):
        out[name + "_packed"], out[name + "_scale"], out[name + "_outer"] = packed_projection(
            experts, rows, columns, g)
    return out


class FakeReader:
    """Per-expert checkpoint tensors under the checkpoint's names."""

    def __init__(self, values, prefix):
        self.values = {}
        for projection in ("gate", "up", "down"):
            for e in range(values["gate_packed"].shape[0]):
                name = f"{prefix}mlp.experts.{e}.{projection}_proj"
                self.values[name + ".weight"] = values[projection + "_packed"][e]
                self.values[name + ".weight_scale"] = values[projection + "_scale"][e]
                self.values[name + ".weight_scale_2"] = values[projection + "_outer"][e]

    def tensor(self, name):
        return self.values[name]

    def dequant(self, prefix, device):
        return dequant_nvfp4(*(self.tensor(prefix + s).to(device)
                               for s in (".weight", ".weight_scale", ".weight_scale_2")))


def default_storage(reader, prefix, experts):
    """The BF16 weights the default build copies in at load (model.py)."""
    gate_up = torch.stack([torch.cat([reader.dequant(f"{prefix}mlp.experts.{e}.gate_proj", "cpu"),
                                      reader.dequant(f"{prefix}mlp.experts.{e}.up_proj", "cpu")], 0)
                           for e in range(experts)])
    down = torch.stack([reader.dequant(f"{prefix}mlp.experts.{e}.down_proj", "cpu") for e in range(experts)])
    return gate_up, down


class NVFP4ShardedExpertTests(unittest.TestCase):
    def setUp(self):
        self.values = tensors()
        self.reader = FakeReader(self.values, "model.language_model.layers.0.")
        self.module = NVFP4Experts.from_reader(self.reader, "model.language_model.layers.0.", 3, "cpu",
                                               optimization="weighted_permutation")
        self.module.grouped_linear = reference_grouped

    def test_expansion_is_bitwise_the_default_storage(self):
        gate_up, down = self.module.materialize()
        ref_gate_up, ref_down = default_storage(self.reader, "model.language_model.layers.0.", 3)
        self.assertTrue(torch.equal(gate_up.view(torch.int16), ref_gate_up.view(torch.int16)))
        self.assertTrue(torch.equal(down.view(torch.int16), ref_down.view(torch.int16)))
        self.assertEqual(self.module.expert_shapes(), (tuple(gate_up.shape), tuple(down.shape)))

    def test_packed_storage_is_frozen_parameters_for_fsdp(self):
        names = {n for n, _ in self.module.named_parameters()}
        self.assertEqual(names, {f"{p}_{part}" for p in ("gate", "up", "down")
                                 for part in ("packed", "scale", "outer")})
        self.assertFalse(any(p.requires_grad for p in self.module.parameters()))
        self.assertEqual(self.module.gate_packed.dtype, torch.uint8)

    def test_forward_and_lora_match_the_default_storage(self):
        torch.manual_seed(3)
        params = install_expert_lora(self.module, dict(rank=4, alpha=8, dropout=0))
        with torch.no_grad():
            for p in params.values():
                p.normal_(0, 0.02)
        gate_up, down = default_storage(self.reader, "model.language_model.layers.0.", 3)
        x = torch.randn(5, 32, dtype=torch.bfloat16, requires_grad=True)
        index = torch.tensor([[0, 2], [1, 2], [0, 1], [2, 0], [1, 0]])
        weights = torch.rand(5, 2, dtype=torch.bfloat16)
        actual = self.module(x, index, weights)
        expected = expert_forward_with_weights(self.module, x, index, weights, gate_up, down,
                                               grouped_linear=reference_grouped,
                                               optimization="weighted_permutation")
        self.assertTrue(torch.equal(actual, expected))
        upstream = torch.randn_like(actual)
        targets = (x, *params.values())
        for a, b in zip(torch.autograd.grad(actual, targets, upstream, retain_graph=True),
                        torch.autograd.grad(expected, targets, upstream)):
            self.assertTrue(torch.equal(a, b))

    def test_rejects_bad_shapes_and_options(self):
        bad = dict(self.values, down_packed=self.values["down_packed"][:, :, :4])
        with self.assertRaises(ValueError):
            NVFP4Experts(bad, optimization="weighted_permutation")
        with self.assertRaises(ValueError):
            NVFP4Experts(self.values, optimization="none")  # the native module serves "none"
        with self.assertRaises(ValueError):
            NVFP4Experts(self.values, optimization="weighted_permutation", materialization="lazy")

    def test_distribution_policy_is_opt_in(self):
        self.assertEqual(DistributionPolicy().expert_storage, "bf16")
        self.assertEqual(DistributionPolicy().nvfp4_materialization, "eager")
        DistributionPolicy(expert_storage="nvfp4_sharded", nvfp4_materialization="fused")
        for bad in (dict(expert_storage="nvfp4"), dict(nvfp4_materialization="fused"),
                    dict(expert_storage="nvfp4_sharded", nvfp4_materialization="lazy")):
            with self.assertRaises(ValueError):
                DistributionPolicy(**bad)

    @unittest.skipUnless(torch.cuda.is_available(), "fused NVFP4 expansion is CUDA-only")
    def test_fused_expansion_is_bitwise_eager(self):
        cuda = {k: v.cuda() for k, v in self.values.items()}
        eager = NVFP4Experts(cuda, optimization="weighted_permutation").materialize()
        fused = NVFP4Experts(cuda, optimization="weighted_permutation", materialization="fused").materialize()
        for a, b in zip(eager, fused):
            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
