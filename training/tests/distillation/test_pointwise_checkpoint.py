"""Fused HC/GDN/MoE adjoints survive packed checkpoint replay and CPU offload."""

import copy
import unittest

import torch

from tests.distillation.test_learner_compile import core
from training.flash.dense_lora import dense_lora_policy
from training.flash.options import FlashOptions
from training.flash.offload import checkpoint_layer

from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen
from training.flash.adapters import FlashDecoderLayer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PointwiseCheckpointTests(unittest.TestCase):
    def test_packed_checkpoint_and_offload(self):
        config = qwen.Qwen4ExpTextConfig(
            hidden_size=2560, hc_count=4, hc_lowrank=320, num_hidden_layers=1,
            num_attention_heads=24, num_key_value_heads=2, head_dim=256,
            moe_intermediate_size=640, shared_expert_intermediate_size=640,
            num_experts=4, num_experts_per_tok=2,
            linear_num_value_heads=3, linear_num_key_heads=1,
            linear_key_head_dim=128, linear_value_head_dim=128,
            layer_types=["linear_attention"], ple_layer_ids=[], output_gate_type="sigmoid",
        )
        config._attn_implementation = "sdpa"
        config._experts_implementation = "grouped_mm"
        for rounding in ("hf", "vllm"):
            torch.manual_seed(149)
            layer = FlashDecoderLayer(config, 0, flex_qsa=False, fp32_router=True,
                                      chunk_rule=core, causal_conv=qwen.causal_conv1d_fn,
                                      fuse_norm_tail=False, options=FlashOptions(
                                          expert_optimization="fused_permutation", hyper_connection="triton_manual",
                                          fuse_gdn_norm=True, fuse_mlp_gates=True, fuse_gdn_qkv=True,
                                          pointwise_rounding=rounding)).cuda().bfloat16()
            for parameter in layer.parameters():
                torch.nn.init.normal_(parameter, std=0.02)
                parameter.requires_grad_(False)
            dense_lora_policy(r=4, alpha=8, fuse_epilogue=True).configure_layer(layer, 0)
            for policy in ("full", "small_gemm"):
                for offload in (0, 1):
                    with self.subTest(rounding=rounding, policy=policy, offload=offload):
                        candidate = copy.deepcopy(layer)
                        wrapped = checkpoint_layer(candidate, 0, offload, checkpoint_policy=policy)
                        x = torch.randn(1, 128, 10240, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        y = x.detach().clone().requires_grad_()
                        kwargs = {"position_embeddings": None, "packed_lengths": (1, 31, 96)}
                        expected, actual = layer(x, **kwargs), wrapped(y, **kwargs)
                        dy = torch.randn_like(expected)
                        expected.backward(dy)
                        actual.backward(dy)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertTrue(torch.equal(x.grad, y.grad))
                        for (_, a), (_, b) in zip(layer.named_parameters(), candidate.named_parameters()):
                            if a.requires_grad:
                                self.assertIsNotNone(a.grad)
                                self.assertTrue(torch.equal(a.grad, b.grad))
                        layer.zero_grad(set_to_none=True)
