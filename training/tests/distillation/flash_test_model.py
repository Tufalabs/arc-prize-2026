"""Random two-layer checkpoint fixture exercising the public Flash builder.

Only the production checkpoint-size assertion is bypassed: construction is
truncated to two layers after validation. Loading, options, LoRA and checkpoint
wrapping still run through build_flash_model. No external weights are needed.
"""

import hashlib
import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from training.flash import adapters
from training.flash.dense_lora import dense_lora_policy
from training.flash.kernel_mode import builder_kwargs, resolve_kernel_stack
from training.flash.model import build_flash_model


# The fast kernel stack (training/flash/kernel_mode.py) of the unwrapped
# fixture, plus selection reuse and attention retention.
FAST_OPTIONS = builder_kwargs(resolve_kernel_stack("fast", packed=True, fsdp=False)) | dict(
    reuse_qsa_selection=True, retain_qsa_attention=True,
)


def test_config(*, ple=False):
    return qwen.Qwen4ExpTextConfig(
        vocab_size=256, hidden_size=2560, hc_count=4, hc_lowrank=320,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        num_attention_heads=24, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128,
        moe_intermediate_size=640, shared_expert_intermediate_size=640,
        num_experts=12, num_experts_per_tok=10,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128,
        indexer_budget=2048, indexer_compress_ratio=4,
        ple_layer_ids=[1] if ple else [], ngram_vocab_size_base=1024,
        eos_token_id=2 if ple else None,
        output_gate_type="sigmoid", use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.25},
    )


def random_tensor(name, shape):
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator) * 0.02
    if name.endswith("linear_attn.norm.weight"):
        value += 1
    return value.bfloat16()


def build_test_model(*, fast=False, lora=True, ple=False, distributed=False, train_routers=False, **options):
    config = test_config(ple=ple)
    config._attn_implementation = "sdpa"
    config._experts_implementation = "grouped_mm"
    with torch.device("meta"):
        template = qwen.Qwen4ExpTextModel(config)
    shapes = {"model.language_model." + name: p.shape
              for name, p in template.named_parameters()}
    shapes["lm_head.weight"] = (config.vocab_size, config.hidden_size)
    buffers = {}
    if ple:
        ngram = template.layers[0].ple.ple_embedding
        prefix = "model.language_model.layers.0.ple.ple_embedding."
        buffers = {
            prefix + "ngram_heads_vocab_sizes": torch.tensor(ngram.head_vocab_sizes),
            prefix + "ngram_heads_offsets": torch.tensor(ngram.head_offsets),
        }

    class RandomReader:
        mapping = buffers

        def __init__(self, root):
            pass

        def tensor(self, name):
            if name in buffers:
                return buffers[name]
            if name.endswith("ngram_embedding.weight_scale"):
                return torch.tensor(1.0, dtype=torch.bfloat16)
            if ".ngram_embedding.shard_" in name:
                rows, width = shapes[name.split(".shard_")[0] + ".weight"]
                return random_tensor(name, (rows // 128, width)).to(torch.float8_e4m3fn)
            return random_tensor(name, shapes[name])

        def dequant(self, name, device):
            shape = ((config.hidden_size, config.moe_intermediate_size)
                     if name.endswith("down_proj") else
                     (config.moe_intermediate_size, config.hidden_size))
            return random_tensor(name, shape).to(device)

    constructor = adapters.FlashTextModel

    def two_layers(full_config, **kwargs):
        full_config.num_hidden_layers = 2
        full_config.layer_types = config.layer_types
        return constructor(full_config, **kwargs)

    kwargs = dict(FAST_OPTIONS) if fast else {}
    kwargs.update(options)
    if lora:
        policy = dense_lora_policy(r=8, alpha=16, fuse_epilogue=fast, train_routers=train_routers)

        def nonzero_lora(layer, index):
            parameters = policy.configure_layer(layer, index)
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(random_tensor(f"lora.{index}.{name}", value.shape).to(value.device))
            return parameters

        kwargs["trainability"] = replace(policy, configure_layer=nonzero_lora)
    stored = config.to_dict()
    stored["num_hidden_layers"] = 48
    stored["layer_types"] = config.layer_types * 24
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps({"text_config": stored}))
        with patch("training.flash.model.FlashTextModel", two_layers):
            return build_flash_model(directory, distributed=distributed, offload_every=0,
                                     reader_factory=RandomReader, **kwargs)


@contextmanager
def snapshot_kernels(*, fixed_config=False):
    # Fix optional FLA autotuning for bitwise regression, including its known
    # four-warp Blackwell recurrence race. Never change the training runtime.
    # FLA ships only in the Linux flash-distillation extra, hence the local import.
    import fla.ops.common.chunk_delta_h as kernels
    from triton.runtime.autotuner import Autotuner

    original = []
    for name in ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
                 "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"):
        # FLA kernel names are looked up by string from the tuple above.
        tuner = getattr(kernels, name)
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        original.append((tuner, tuner.configs))
        tuner.configs = [config for config in tuner.configs if config.num_warps == 2]
        if fixed_config:
            # One configuration removes the remaining node-dependent tuning.
            tuner.configs = [config for config in tuner.configs
                             if config.kwargs == {"BV": 32} and config.num_stages == 2]
            assert len(tuner.configs) == 1
        assert tuner.configs
        tuner.cache.clear()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        for tuner, configs in original:
            tuner.configs = configs
            tuner.cache.clear()
