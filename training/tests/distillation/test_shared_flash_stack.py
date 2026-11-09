import inspect
import unittest
from pathlib import Path

import torch

from training.flash.model import FlashModelBundle
from training.flash.policies import DistributionPolicy, TrainabilityPolicy


class SharedFlashStackTests(unittest.TestCase):
    def test_policy_defaults_reproduce_distillation_shape(self):
        trainability = TrainabilityPolicy.router_only()
        distribution = DistributionPolicy()
        self.assertEqual(trainability.mode, "routers")
        self.assertEqual(trainability.router_backend, "fp32_reference")
        self.assertTrue(trainability.training)
        self.assertEqual(
            (distribution.reshard_after_forward, distribution.forward_prefetch,
             distribution.backward_prefetch),
            (True, 0, 0),
        )

    def test_invalid_policy_combinations_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "configure_layer"):
            TrainabilityPolicy(mode="custom")
        with self.assertRaisesRegex(ValueError, "FP32 reference"):
            TrainabilityPolicy(mode="routers", router_backend="stock")
        # Experts are FSDP-sharded: no strategy or NVFP4 layout choice; storage is the opt-in
        # expert_storage ("bf16" default, "nvfp4_sharded"), which refuses other values.
        for retired in ("strategy", "nvfp4_distribution"):
            with self.assertRaises(TypeError):
                DistributionPolicy(**{retired: "fsdp"})
        with self.assertRaises(ValueError):
            DistributionPolicy(expert_storage="fsdp")

    def test_custom_policy_is_an_explicit_pre_wrap_adapter_seam(self):
        adapter = torch.nn.Parameter(torch.ones(3))

        def configure_layer(layer, index):
            self.assertIsInstance(layer, torch.nn.Module)
            return {f"adapter_{index}": adapter}

        policy = TrainabilityPolicy(
            mode="custom",
            router_backend="stock",
            configure_layer=configure_layer,
        )
        configured = policy.configure_layer(torch.nn.Identity(), 7)
        self.assertEqual(set(configured), {"adapter_7"})
        self.assertIs(configured["adapter_7"], adapter)

    def test_bundle_exposes_named_parameter_groups(self):
        router = torch.nn.Parameter(torch.ones(2))
        frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        bundle = FlashModelBundle(
            torch.nn.Identity(),
            torch.nn.Identity(),
            {"routers": {"router": router}, "custom": {"frozen": frozen}},
        )
        self.assertEqual(set(bundle.trainable), {"router"})
        self.assertIs(bundle.trainable["router"], router)

    def test_distillation_adapter_delegates_to_shared_builder(self):
        from sft.distillation import flash_model
        from training.flash import model

        self.assertIs(flash_model.build_flash_model, model.build_flash_model)

    def test_shared_package_has_no_distillation_dependency(self):
        package = Path(inspect.getfile(FlashModelBundle)).parent
        offenders = []
        for source in package.glob("*.py"):
            if "sft.distillation" in source.read_text():
                offenders.append(source.name)
        self.assertEqual(offenders, [])

    def test_transformers_adapters_are_explicit_and_do_not_mutate_globals(self):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen
        from transformers.models.qwen4_exp.configuration_qwen4_exp import (
            Qwen4ExpTextConfig,
        )

        from training.flash.adapters import (
            FlashTextModel,
            FP32TopKRouter,
            QualifiedGatedDeltaNet,
            validate_transformers_contract,
        )
        from training.flash.batched_qsa import BatchedQSAIndexer
        from training.flash.flex_qsa import FlexQSAAttention

        validate_transformers_contract()
        config = Qwen4ExpTextConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            num_experts=12,
            num_experts_per_tok=3,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=8,
            indexer_compress_ratio=2,
            layer_types=["full_attention", "linear_attention"],
            ple_layer_ids=[],
        )
        config._attn_implementation = "sdpa"
        config._experts_implementation = "grouped_mm"
        original_chunk_rule = qwen.torch_chunk_gated_delta_rule

        def chunk_rule(*args, **kwargs):
            return original_chunk_rule(*args, **kwargs)

        with torch.device("meta"):
            model = FlashTextModel(
                config,
                flex_qsa=True,
                fp32_router=True,
                chunk_rule=chunk_rule,
                causal_conv=qwen.causal_conv1d_fn,
                fuse_norm_tail=False,
            )
        self.assertIsInstance(model.layers[0].self_attn, FlexQSAAttention)
        self.assertIsInstance(model.layers[1].linear_attn, QualifiedGatedDeltaNet)
        self.assertTrue(
            all(isinstance(layer.mlp.gate, FP32TopKRouter) for layer in model.layers)
        )
        self.assertIs(qwen.torch_chunk_gated_delta_rule, original_chunk_rule)
        with torch.device("meta"):
            sdpa_model = FlashTextModel(
                config,
                flex_qsa=False,
                fp32_router=True,
                chunk_rule=chunk_rule,
                causal_conv=qwen.causal_conv1d_fn,
                fuse_norm_tail=False,
            )
        self.assertIsInstance(sdpa_model.layers[0].self_attn.indexer, BatchedQSAIndexer)

    def test_shared_package_has_no_runtime_monkeypatching(self):
        package = Path(inspect.getfile(FlashModelBundle)).parent
        forbidden = ("SOURCE_SHA", "__globals__", "types.MethodType")
        offenders = {
            source.name: token
            for source in package.glob("*.py")
            for token in forbidden
            if token in source.read_text()
        }
        self.assertEqual(offenders, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
