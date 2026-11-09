"""Fail before work for unsupported public Flash option/input combinations."""

import json
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from tests.distillation.flash_test_model import build_test_model, test_config
from training.flash import adapters
from training.flash.adapters import (
    FlashDecoderLayer,
    QualifiedGatedDeltaNet,
    StockTopKRouter,
    _packed_boundaries,
)
from training.flash.batched_qsa import BatchedQSAIndexer
from training.flash.decision_replay import DecisionTape
from training.flash.dense_lora import dense_lora_policy
from training.flash.flex_qsa import FlexQSAAttention
from training.flash.head_loss import CachedHeadCE, ChunkedCE
from training.flash.gated_pointwise import TritonGatedNorm
from training.flash.hyper_connection import grouped_norm, residual_inject, residual_inject_norm, sigmoid_mean
from training.flash.model import build_flash_model
from training.flash.moe_permutation import combine_experts, gather_tokens
from training.flash.offload import checkpoint_layer
from training.flash.options import FlashOptions
from training.flash.selective_checkpoint import selective_contexts
from training.flash.selector_scoring import bf16_scores, compiled_fp32_scores, fp32_scores


class APIContractTests(unittest.TestCase):
    def test_every_flash_option_validates_direct_construction(self):
        for field in fields(FlashOptions):
            with self.subTest(field=field.name), self.assertRaisesRegex(ValueError, field.name):
                FlashOptions(**{field.name: "typo"})
        for name in ("gdn_grouped_heads", "fuse_gdn_norm", "fuse_mlp_gates", "fuse_gdn_qkv", "fuse_qsa_norm"):
            for value in (0, 1, None, "false"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    FlashOptions(**{name: value})
        with self.assertRaisesRegex(ValueError, "test_gdn_grouped_heads.py"):
            FlashOptions(gdn_grouped_heads=True)

    def test_complete_serving_paths_required(self):
        for hc in ("none", "triton", "triton_inject_norm", "triton_manual"):
            for fuse in (False, True):
                kwargs = dict(pointwise_rounding="vllm", hyper_connection=hc, fuse_gdn_norm=fuse)
                if fuse and hc != "none":
                    FlashOptions(**kwargs)
                else:
                    with self.assertRaisesRegex(ValueError, "GDN gated norm and HC"):
                        FlashOptions(**kwargs)

    def test_builder_scalars_rejected_before_import_or_checkpoint(self):
        invalid = {field.name: "typo" for field in fields(FlashOptions)}
        invalid.update(device="typo", role="typo", attention_backend="typo", compile_precision="typo",
                       checkpoint_policy="typo", offload_every=-1,
                       distributed="false", reuse_qsa_selection="false", retain_qsa_attention="false",
                       fuse_norm_tail="false", compile_layers="false", replicate_frozen_routers="false")
        reader = Mock(side_effect=AssertionError("reader called"))
        for name, value in invalid.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                build_flash_model("/does-not-exist", reader_factory=reader, **{name: value})
        with self.assertRaisesRegex(ValueError, "replicate_frozen_routers=True requires FSDP"):
            build_flash_model("/does-not-exist", reader_factory=reader, distributed=False,
                              replicate_frozen_routers=True)
        reader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "test_gdn_grouped_heads.py"):
            build_flash_model("/does-not-exist", gdn_grouped_heads=True)

    def test_group_sparse_requires_the_specialized_indexer_geometry(self):
        config = test_config()
        config.indexer_compress_ratio = 2
        with torch.device("meta"):
            FlexQSAAttention(config, 1, qsa_kernel="physical_sparse")
            with self.assertRaisesRegex(ValueError, "group_sparse.*compress_ratio=2, block_topk=1024"):
                FlexQSAAttention(config, 1, qsa_kernel="group_sparse")

    @unittest.skipUnless(torch.cuda.is_available(), "FLA runtime required")
    def test_builder_config_counts_raise_explicitly(self):
        for counts in ((47, 12, 10), (48, 9, 10), (48, 12, 9)):
            config = SimpleNamespace(num_hidden_layers=counts[0], num_experts=counts[1], num_experts_per_tok=counts[2])
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "config.json").write_text(json.dumps({"text_config": {}}))
                with patch("training.flash.model.Qwen4ExpTextConfig", return_value=config):
                    with self.assertRaisesRegex(ValueError, f"num_hidden_layers={counts[0]}, num_experts={counts[1]}, num_experts_per_tok={counts[2]}"):
                        build_flash_model(directory, reader_factory=Mock(), distributed=False)

    def test_sdpa_tape_rejected_before_selection_including_fallback(self):
        indexer = BatchedQSAIndexer.__new__(BatchedQSAIndexer)
        nn.Module.__init__(indexer)
        for scope in (DecisionTape().record, DecisionTape().replay):
            with scope(), patch("training.flash.batched_qsa._causal_layout") as layout:
                with self.assertRaisesRegex(RuntimeError, "attention_backend='sdpa'"):
                    indexer(None, None, None)
                layout.assert_not_called()

    def test_stock_router_rejects_active_tape(self):
        config = qwen.Qwen4ExpTextConfig(
            hidden_size=16, num_hidden_layers=4, num_experts=8, num_experts_per_tok=2,
            moe_intermediate_size=8, shared_expert_intermediate_size=8, ple_layer_ids=[],
        )
        with torch.device("meta"):
            for fp32_router, router in ((False, StockTopKRouter), (True, adapters.FP32TopKRouter)):
                layer = FlashDecoderLayer(
                    config, 0, flex_qsa=False, fp32_router=fp32_router, chunk_rule=None,
                    causal_conv=None, fuse_norm_tail=False,
                )
                self.assertIs(type(layer.mlp.gate), router)
        router = StockTopKRouter(config)
        nn.init.normal_(router.weight)
        x = torch.randn(5, 16)
        for actual, expected in zip(router(x), qwen.Qwen4ExpTextTopKRouter.forward(router, x)):
            self.assertTrue(torch.equal(actual, expected))
        for scope in (DecisionTape().record, DecisionTape().replay):
            with scope(), self.assertRaisesRegex(RuntimeError, "router_backend='stock'"):
                router(x)

    def test_gated_norm_requires_gate(self):
        with self.assertRaisesRegex(ValueError, "gate is required"):
            TritonGatedNorm(8)(torch.randn(2, 8))

    def test_inference_mode_tape_entry_rejected(self):
        tape = DecisionTape()
        with torch.inference_mode():
            for scope in (tape.record, tape.replay):
                with self.assertRaisesRegex(RuntimeError, "inference_mode=True"):
                    with scope():
                        self.fail("entered inference tape")
        self.assertFalse(tape.entries)

    def test_head_shapes_dtypes_and_frozen_weights(self):
        hidden, weight, targets = torch.randn(4, 8), torch.randn(13, 8), torch.zeros(4, dtype=torch.int64)
        cases = ((hidden, weight, targets[:2]), (hidden, weight, targets[:, None]),
                 (hidden, weight, targets.float()), (hidden, weight[:, :4], targets),
                 (hidden[None], weight, targets), (hidden, weight[None], targets),
                 (hidden, weight.requires_grad_(), targets))
        for op in (ChunkedCE, CachedHeadCE):
            for args in cases:
                with self.subTest(op=op.__name__, shapes=[v.shape for v in args]):
                    with self.assertRaises((ValueError, TypeError)):
                        op.apply(*args, 2)

    def test_small_gemm_reuse_rejected_and_repeated_backward_named(self):
        layer = nn.Linear(4, 4)
        with self.assertRaisesRegex(ValueError, "small_gemm.*reuse_qsa_selection=True"):
            checkpoint_layer(layer, 0, 0, reuse_qsa_selection=True, checkpoint_policy="small_gemm")
        x = torch.randn(3, 4, requires_grad=True)
        out = checkpoint(lambda x: layer(x).sin(), x, use_reentrant=False, context_fn=selective_contexts)
        out.sum().backward(retain_graph=True)
        with self.assertRaisesRegex(RuntimeError, "small_gemm.*repeated backward.*retain_graph=True"):
            out.sum().backward()

    def test_dense_lora_global_target_census(self):
        policy = dense_lora_policy(target_modules=["q_proj", "out_proj"])
        layer = nn.Module()
        layer.linear_attn = nn.Module()
        layer.linear_attn.out_proj = nn.Linear(8, 8)
        trainable = policy.configure_layer(layer, 0)
        self.assertTrue(trainable)
        with self.assertRaisesRegex(ValueError, "matched nowhere.*q_proj"):
            policy.validate_trainables({f"layers.0.{name}": p for name, p in trainable.items()})
        empty = dense_lora_policy(target_modules=["q_proj"])
        with self.assertRaisesRegex(ValueError, "matched nowhere.*q_proj"):
            empty.validate_trainables(empty.configure_layer(layer, 0))

    def test_packed_gdn_boundary_provenance_before_projection(self):
        layer = QualifiedGatedDeltaNet.__new__(QualifiedGatedDeltaNet)
        nn.Module.__init__(layer)
        layer.options = FlashOptions(gdn_convolution="fla")
        x = torch.randn(1, 128, 8)
        cu = torch.tensor([0, 128], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "may not supply both"):
            layer(x, packed_lengths=(1, 127), cu_seq_lens_q=cu)
        trusted = _packed_boundaries((1, 127), None, 128, x.device)
        self.assertEqual(trusted.tolist(), [0, 1, 128])
        self.assertIs(_packed_boundaries((1, 127), trusted, 128, x.device), trusted)
        trusted[1] = 2
        with self.assertRaisesRegex(ValueError, "may not supply both"):
            _packed_boundaries((1, 127), trusted, 128, x.device)
        layer.conv1d = SimpleNamespace(weight=torch.ones(8, 1, 4), bias=None)
        layer.activation = "silu"
        with patch("training.flash.adapters.fla_convolution", return_value=x) as conv:
            layer._convolution(x.transpose(1, 2), (1, 127))
        self.assertEqual(conv.call_args.args[-1].tolist(), [0, 1, 128])

    def test_permutation_metadata_shapes_and_dtypes(self):
        x, w, order = torch.randn(2, 8), torch.randn(2, 3), torch.arange(6)
        for bad in (order[:5], order[:, None], order.float()):
            with self.assertRaises((ValueError, TypeError)):
                gather_tokens(x, bad, 3)
            with self.assertRaises((ValueError, TypeError)):
                combine_experts(x.repeat_interleave(3, 0), w, bad)
        for k in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "top_k"):
                gather_tokens(x, order, k)
        with self.assertRaisesRegex(ValueError, "weights"):
            combine_experts(x, w, order)
        # The former contradictory-inverse API is no longer accepted.
        with self.assertRaises(TypeError):
            gather_tokens(x, order, order.flip(0), 3)
        with self.assertRaises(TypeError):
            combine_experts(x.repeat_interleave(3, 0), w, order, order.flip(0))

    def test_fp32_selectors_ignore_autocast(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            q = torch.randn(13, 4, 32, device=device, dtype=torch.bfloat16)
            k = torch.randn(19, 32, device=device, dtype=torch.bfloat16)
            for call in (fp32_scores, compiled_fp32_scores):
                expected = call(q, k)
                with torch.autocast(device, dtype=torch.bfloat16):
                    actual = call(q, k)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_builder_rejects_dense_lora_matching_no_layers(self):
        constructor = adapters.FlashTextModel
        def only_gdn(config, **kwargs):
            model = constructor(config, **kwargs)
            model.layers = model.layers[:1]
            return model
        with patch.object(adapters, "FlashTextModel", only_gdn):
            # A partial match still trains out_proj; the policy alone must catch q_proj.
            with self.assertRaisesRegex(ValueError, "matched nowhere.*q_proj"):
                build_test_model(lora=False, trainability=dense_lora_policy(target_modules=["q_proj", "out_proj"]))
            unchecked = replace(dense_lora_policy(target_modules=["q_proj"]), validate_trainables=None)
            with self.assertRaisesRegex(ValueError, "custom trainability trained no parameters"):
                build_test_model(lora=False, trainability=unchecked)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_hyper_connection_and_selector_shapes_fail_before_allocation(self):
        x = torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16)
        h, weights, gain = x[..., :16], x[..., :4], x.new_zeros(64)
        calls = [lambda: grouped_norm(x, gain[:16], 4, 1e-6),
                 lambda: grouped_norm(x, gain, 0, 1e-6),
                 lambda: grouped_norm(x[..., :63], gain, 4, 1e-6),
                 lambda: sigmoid_mean(x, x[:, :1], 4),
                 lambda: residual_inject(x, h[:, :1], weights),
                 lambda: residual_inject(x, h, weights[..., :3]),
                 lambda: residual_inject_norm(x, h, weights, gain[:16], 1e-6),
                 lambda: bf16_scores(x, x[0, :, :32]),
                 lambda: bf16_scores(x[0], x[0]),
                 lambda: bf16_scores(x, x)]
        for call in calls:
            with self.assertRaises(ValueError):
                call()
