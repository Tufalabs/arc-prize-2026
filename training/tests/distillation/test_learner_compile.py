import copy
import unittest

import torch

from training.flash.dense_lora import dense_lora_policy
from training.flash.options import FlashOptions
from training.flash.offload import checkpoint_layer

from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen
from training.flash.adapters import FlashDecoderLayer
from training.flash.compile import compile_decoder


def core(query, key, value, **kwargs):
    return query + key + value, None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecoderCompileTests(unittest.TestCase):
    def test_packed_checkpointed_decoder_gradient_parity(self):
        config = qwen.Qwen4ExpTextConfig(
            hidden_size=16, hc_count=2, hc_lowrank=8, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2,
            linear_num_value_heads=1, linear_num_key_heads=1,
            linear_key_head_dim=8, linear_value_head_dim=8,
            indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=8,
            indexer_budget=8, indexer_compress_ratio=2,
            layer_types=["linear_attention"], ple_layer_ids=[],
        )
        config._attn_implementation = "sdpa"
        config._experts_implementation = "grouped_mm"
        # Compile fused first, then unfused: reused resume graphs must guard
        # the constructor choice instead of calling a stale optional function.
        for fuse_gate in (True, False):
            torch.manual_seed(13)
            layer = FlashDecoderLayer(config, 0, flex_qsa=False, fp32_router=True,
                                      chunk_rule=core, causal_conv=qwen.causal_conv1d_fn,
                                      fuse_norm_tail=False,
                                      options=FlashOptions(expert_optimization="fused_permutation",
                                                           fuse_mlp_gates=fuse_gate)).cuda().bfloat16()
            for parameter in layer.parameters():
                torch.nn.init.normal_(parameter, std=0.02)
                parameter.requires_grad_(False)
            dense_lora_policy(r=4, alpha=8).configure_layer(layer, 0)
            for precision in ("reference", "fused"):
                for every in (0, 1):
                    compiled = copy.deepcopy(layer)
                    compile_decoder(compiled, precision=precision)
                    wrapped = checkpoint_layer(compiled, 0, every, checkpoint_policy="small_gemm")
                    for n in (128, 256):
                        x = torch.randn(1, n, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                        y = x.detach().clone().requires_grad_()
                        kwargs = {"position_embeddings": None, "packed_lengths": (n // 2, n // 2)}
                        reference = layer(x, **kwargs)
                        actual = wrapped(y, **kwargs)
                        upstream = torch.randn_like(reference)
                        reference.backward(upstream)
                        actual.backward(upstream)
                        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
                        torch.testing.assert_close(x.grad, y.grad, atol=0.02, rtol=0.02)
                        for (_, a), (_, b) in zip(layer.named_parameters(), compiled.named_parameters()):
                            if a.grad is not None:
                                torch.testing.assert_close(a.grad, b.grad, atol=0.03, rtol=0.03)
                        layer.zero_grad(set_to_none=True)
                        compiled.zero_grad(set_to_none=True)
