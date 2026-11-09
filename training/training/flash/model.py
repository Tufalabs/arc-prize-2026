"""Trainer-independent checkpoint-backed Qwen3.8-Flash model stack."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from .adapters import FlashTextModel, validate_transformers_contract
from .adapters import router_forward as adapter_router_forward
from .compile import compile_decoder
from .policies import DistributionPolicy, TrainabilityPolicy
from .reader import Reader
from .options import FlashOptions
from .offload import checkpoint_layer, resolve_qsa_retention, validate_offload_chunk
from .expert_lora import install_expert_lora
from .nvfp4_experts import NVFP4Experts
from .qsa_policy import validate_qsa_kernel

class HostEmbedding(nn.Module):
    """Read-only file-backed FP8 table; gather bytes on CPU, then scale in BF16.

    No registered tensor: FSDP never moves or shards the host table. The native
    NGramEmbedding consults .weight.device; this zero-size property is CPU-only.
    All ranks map the same files and share Linux page cache, not Python copies.
    """

    def __init__(self, reader, prefix):
        super().__init__()
        self.reader, self.prefix = reader, prefix
        self.parts = [reader.tensor(f"{prefix}.shard_{i}.weight") for i in range(128)]
        self.ends = torch.tensor([p.shape[0] for p in self.parts]).cumsum(0)
        self.scale = reader.tensor(prefix + ".weight_scale").clone()
        assert all(p.dtype == torch.float8_e4m3fn for p in self.parts)
        assert (
            self.scale.numel() == 1
            and torch.isfinite(self.scale).all()
            and self.scale.item() > 0
        )

    @property
    def weight(self):
        return torch.empty(0, device="cpu")

    def forward(self, ids):
        ids = ids.cpu()
        flat = ids.reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.ends[-1]):
            raise ValueError("Ngram ID out of range")
        owners = torch.bucketize(flat, self.ends, right=True)
        result = torch.empty(
            (flat.numel(), self.parts[0].shape[1]), dtype=torch.bfloat16
        )
        for i in owners.unique().tolist():
            positions = (owners == i).nonzero().flatten()
            local = flat[positions] - (0 if i == 0 else self.ends[i - 1])
            values = (
                self.parts[i]
                .view(torch.uint8)
                .index_select(0, local)
                .view(torch.float8_e4m3fn)
            )
            result[positions] = values.bfloat16() * self.scale
        return result.reshape(*ids.shape, -1)


def router_forward(self, hidden_states):
    # Legacy callers can supply native HF routers without a layer_idx attribute.
    return adapter_router_forward(self, hidden_states, getattr(self, "layer_idx", None))


def assign(module, name, tensor, parameter=True):
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    if parameter:
        old = getattr(parent, leaf)
        if old.shape != tensor.shape:
            raise ValueError(f"Shape mismatch: {name}: {old.shape} vs {tensor.shape}")
        setattr(parent, leaf, nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(parent, leaf, tensor)


@dataclass
class FlashModelBundle:
    model: nn.Module
    head: nn.Module
    parameter_groups: dict[str, dict[str, nn.Parameter]]

    @property
    def trainable(self):
        return {
            name: parameter
            for group in self.parameter_groups.values()
            for name, parameter in group.items()
            if parameter.requires_grad
        }


def build_flash_model(
    root,
    *,
    role="student",
    trainability=None,
    distribution=None,
    reader_factory: Callable[[Path], Reader] = Reader,
    distributed=True,
    device="cuda",
    offload_every=1,
    offload_chunk_mib=None,
    attention_backend="sdpa",
    qsa_kernel=None,
    reuse_qsa_selection=False,
    retain_qsa_attention=None,
    expert_optimization="weighted_permutation",
    fuse_norm_tail=True,
    gdn_convolution="native",
    ple_convolution="native",
    selector_policy="batched_fp32_top513_v1",
    compile_layers=False,
    compile_precision="reference",
    checkpoint_policy="full",
    hyper_connection="none",
    gdn_grouped_heads=False,
    fuse_gdn_norm=False,
    fuse_mlp_gates=False,
    fuse_gdn_qkv=False,
    fuse_qsa_norm=False,
    pointwise_rounding="hf",
    replicate_frozen_routers=False,
):
    options = FlashOptions(
        gdn_convolution=gdn_convolution, ple_convolution=ple_convolution,
        selector_policy=selector_policy, qsa_kernel=qsa_kernel,
        expert_optimization=expert_optimization, hyper_connection=hyper_connection,
        gdn_grouped_heads=gdn_grouped_heads, fuse_gdn_norm=fuse_gdn_norm,
        fuse_mlp_gates=fuse_mlp_gates, fuse_gdn_qkv=fuse_gdn_qkv,
        fuse_qsa_norm=fuse_qsa_norm, pointwise_rounding=pointwise_rounding,
    )
    for name, value, allowed in (
        ("role", role, ("student", "teacher")),
        ("attention_backend", attention_backend, ("sdpa", "flex_qsa")),
        ("compile_precision", compile_precision, ("reference", "fused")),
        ("checkpoint_policy", checkpoint_policy, ("full", "small_gemm")),
    ):
        if value not in allowed:
            raise ValueError(f"build_flash_model: {name}={value!r}; allowed values are {allowed}")
    for name, value in (
        ("distributed", distributed), ("reuse_qsa_selection", reuse_qsa_selection),
        ("fuse_norm_tail", fuse_norm_tail), ("compile_layers", compile_layers),
        ("replicate_frozen_routers", replicate_frozen_routers),
    ):
        if type(value) is not bool:
            raise ValueError(f"build_flash_model: {name}={value!r}; allowed values are False, True (bool)")
    if type(offload_every) is not int or offload_every < 0:
        raise ValueError(f"build_flash_model: offload_every={offload_every!r}; allowed values are nonnegative integers")
    validate_offload_chunk(offload_chunk_mib)
    if retain_qsa_attention is not None and type(retain_qsa_attention) is not bool:
        raise ValueError(f"build_flash_model: retain_qsa_attention={retain_qsa_attention!r}; allowed values are None, False, True")
    try:
        torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"build_flash_model: device={device!r}; requires a torch device such as 'cpu' or 'cuda'") from exc
    qsa_kernel = validate_qsa_kernel(qsa_kernel, attention_backend)
    options = replace(options, qsa_kernel=qsa_kernel)
    teacher = role == "teacher"
    trainability = trainability or (
        TrainabilityPolicy.frozen_reference()
        if teacher
        else TrainabilityPolicy.router_only()
    )
    distribution = distribution or DistributionPolicy()
    if teacher and trainability.mode != "frozen":
        raise ValueError("teacher models must be frozen")
    if teacher and distribution != DistributionPolicy():
        raise ValueError("teacher distribution must keep the default FSDP settings")
    # A distributed build always shards with FSDP; routed experts are BF16
    # FSDP parameters like the other frozen weights.
    use_fsdp = distributed
    if replicate_frozen_routers and not use_fsdp:
        raise ValueError("build_flash_model: replicate_frozen_routers=True requires FSDP (distributed=True)")
    if teacher and attention_backend != "sdpa":
        raise ValueError(f"build_flash_model: attention_backend={attention_backend!r}; teacher allows only 'sdpa'")
    retain_qsa_attention = resolve_qsa_retention(
        attention_backend, reuse_qsa_selection, retain_qsa_attention
    )
    if retain_qsa_attention and (
        teacher or attention_backend != "flex_qsa" or not reuse_qsa_selection
    ):
        raise ValueError("QSA retention requires student FlexQSA and selection reuse")
    if reuse_qsa_selection and (teacher or attention_backend != "flex_qsa"):
        raise ValueError("QSA selection reuse requires student flex_qsa")
    if trainability.expert_lora and expert_optimization == "none":
        raise ValueError("expert LoRA requires an optimized expert implementation")
    if distribution.expert_storage == "nvfp4_sharded" and (teacher or expert_optimization == "none"):
        raise ValueError("expert_storage='nvfp4_sharded' requires a student with an optimized expert implementation")
    # FLA ships only in the Linux flash-distillation extra.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    validate_transformers_contract()
    reader = reader_factory(Path(root))
    config = Qwen4ExpTextConfig(
        **json.loads((Path(root) / "config.json").read_text())["text_config"]
    )
    config._attn_implementation = "sdpa"
    # FlashExperts dispatches its own optimized paths; the native registry
    # entry only serves expert_optimization="none".
    config._experts_implementation = "grouped_mm"
    config.use_cache = False
    if not (
        config.num_hidden_layers == 48
        and (config.num_experts == 307 if teacher else 10 <= config.num_experts <= 307)
        and config.num_experts_per_tok == 10
    ):
        raise ValueError(
            f"build_flash_model: num_hidden_layers={config.num_hidden_layers}, "
            f"num_experts={config.num_experts}, num_experts_per_tok={config.num_experts_per_tok}; "
            "requires 48 layers, 10 experts per token, and "
            + ("307 teacher experts" if teacher else "10..307 student experts")
        )
    with torch.device("meta"):
        model = FlashTextModel(
            config,
            flex_qsa=attention_backend == "flex_qsa",
            fp32_router=trainability.router_backend == "fp32_reference",
            chunk_rule=chunk_gated_delta_rule,
            causal_conv=qwen.causal_conv1d_fn,
            fuse_norm_tail=fuse_norm_tail and trainability.training,
            options=options,
        )
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    parameter_groups = {"routers": {}, "custom": {}, "expert_lora": {}}
    frozen_router_replicas = set()
    routers = parameter_groups["routers"]
    prefix = "model.language_model."
    for i, layer in enumerate(model.layers):
        lp = f"{prefix}layers.{i}."
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = HostEmbedding(
                reader, lp + "ple.ple_embedding.ngram_embedding"
            )
        packed = distribution.expert_storage == "nvfp4_sharded"
        if packed:
            # Opt-in: the checkpoint's packed experts as frozen parameters (nvfp4_experts.py).
            experts = layer.mlp.experts
            layer.mlp.experts = NVFP4Experts.from_reader(
                reader, lp, config.num_experts, device, optimization=experts.optimization,
                fuse_gate=experts.fuse_gate, materialization=distribution.nvfp4_materialization,
            )
        for name, p in list(layer.named_parameters()):
            if packed and name.startswith("mlp.experts."):
                continue
            if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                assign(
                    layer,
                    name,
                    torch.empty(p.shape, device=device, dtype=torch.bfloat16),
                )
            else:
                assign(layer, name, reader.tensor(lp + name).to(device))
        # The checkpoint's NVFP4 experts, expanded once to BF16 at load (default storage).
        with torch.no_grad():
            for e in range(0 if packed else config.num_experts):
                ep = lp + f"mlp.experts.{e}."
                layer.mlp.experts.gate_up_proj[e].copy_(
                    torch.cat(
                        [
                            reader.dequant(ep + "gate_proj", device),
                            reader.dequant(ep + "up_proj", device),
                        ],
                        0,
                    )
                )
                layer.mlp.experts.down_proj[e].copy_(
                    reader.dequant(ep + "down_proj", device)
                )
        for name, b in list(layer.named_buffers()):
            if lp + name in reader.mapping:
                assign(layer, name, reader.tensor(lp + name).to(device), False)
            elif name.endswith("layer_multipliers"):
                ng = layer.ple.ple_embedding
                assign(
                    layer,
                    name,
                    qwen._build_layer_multipliers(
                        ng.unigram_vocab_size,
                        ng.ngram_size,
                        ng.ple_layer_index,
                        ng.seed,
                    ).to(device),
                    False,
                )
            elif not b.is_meta:
                assign(layer, name, b.to(device), False)
            else:
                raise ValueError("Unresolved buffer " + lp + name)
        gate = layer.mlp.gate
        layer_trainables = {}
        if trainability.router_backend == "fp32_reference":
            gate.weight = nn.Parameter(
                gate.weight.float(), requires_grad=trainability.routers_trainable
            )
        else:
            gate.weight.requires_grad_(False)
        if trainability.routers_trainable:
            name = f"layers.{i}.mlp.gate.weight"
            routers[name] = gate.weight
            layer_trainables[name] = gate.weight
        if trainability.mode == "custom":
            configured = trainability.configure_layer(layer, i)
            if not isinstance(configured, dict) or not all(
                isinstance(name, str) and isinstance(parameter, nn.Parameter)
                for name, parameter in configured.items()
            ):
                raise TypeError("configure_layer must return named Parameters")
            for name, parameter in configured.items():
                if not parameter.requires_grad:
                    raise ValueError(f"custom trainable is frozen: {name}")
                qualified = f"layers.{i}.{name}"
                parameter_groups["custom"][qualified] = parameter
                layer_trainables[qualified] = parameter
        if trainability.expert_lora:
            adapters = install_expert_lora(layer.mlp.experts, trainability.expert_lora)
            for name, parameter in adapters.items():
                qualified = f"layers.{i}.mlp.experts.{name}"
                parameter_groups["expert_lora"][qualified] = parameter
                layer_trainables[qualified] = parameter
        if compile_layers:
            compile_decoder(layer, precision=compile_precision)
        layer.train()
        wrapped = (
            layer
            if teacher
            else checkpoint_layer(
                layer,
                i,
                offload_every,
                reuse_qsa_selection=reuse_qsa_selection and not hasattr(layer, "linear_attn"),
                retain_qsa_attention=retain_qsa_attention and not hasattr(layer, "linear_attn"),
                # QSA replay deliberately skips selector/retained-core ops;
                # SAC's per-op FIFO cannot cache across those asymmetric paths.
                checkpoint_policy=(checkpoint_policy if hasattr(layer, "linear_attn") else "full"),
                offload_chunk_mib=offload_chunk_mib,
            )
        )
        if use_fsdp:
            # Trainables, trainable routers included, stay replicated outside
            # FSDP; replicate_frozen_routers does the same for frozen ones.
            ignored_params = set(layer_trainables.values())
            if replicate_frozen_routers and not gate.weight.requires_grad:
                ignored_params.add(gate.weight)
                frozen_router_replicas.add(gate.weight)
            fully_shard(
                wrapped,
                reshard_after_forward=distribution.reshard_after_forward,
                ignored_params=ignored_params,
            )
        model.layers[i] = wrapped
        print(json.dumps({"loaded_layer": i}), flush=True)
    if trainability.mode == "custom":
        if trainability.validate_trainables is not None:
            trainability.validate_trainables(parameter_groups["custom"])
        if not parameter_groups["custom"]:
            raise ValueError("build_flash_model: custom trainability trained no parameters")
    if use_fsdp:
        for i, wrapped in enumerate(model.layers):
            if distribution.forward_prefetch:
                wrapped.set_modules_to_forward_prefetch(
                    list(model.layers[i + 1 : i + 1 + distribution.forward_prefetch])
                )
            if distribution.backward_prefetch:
                wrapped.set_modules_to_backward_prefetch(
                    list(reversed(model.layers[max(0, i - distribution.backward_prefetch) : i]))
                )
    for name in ("embed_tokens", "hyper_connection_mixer"):
        module = getattr(model, name)
        for pn, _ in list(module.named_parameters()):
            assign(module, pn, reader.tensor(prefix + name + "." + pn).to(device))
        if use_fsdp:
            fully_shard(module, reshard_after_forward=True)
    model.rotary_emb = qwen.Qwen4ExpTextRotaryEmbedding(config=config).to(device)
    assign(head, "weight", reader.tensor("lm_head.weight").to(device))
    # Head stays replicated (1.18 GiB/rank), so chunked custom backward can use it.
    if use_fsdp:
        fully_shard(
            model,
            reshard_after_forward=True,
            ignored_params=set(parameter_groups["routers"].values())
            | set(parameter_groups["custom"].values())
            | set(parameter_groups["expert_lora"].values())
            | frozen_router_replicas,
        )
        register_fsdp_forward_method(model, "embed_input_ids")
    if trainability.training:
        if fuse_norm_tail and any(
            "norm" in name for name in parameter_groups["custom"]
        ):
            raise ValueError("fused frozen norm cannot train norm parameters")
    model.train(trainability.training)
    assert all(not p.is_meta for p in model.parameters())
    return FlashModelBundle(model, head, parameter_groups)


def frozen_fingerprint(model, head):
    hashes = {}
    for name, tensor in (
        list(model.named_parameters())
        + [("lm_head.weight", head.weight)]
        + list(model.named_buffers())
    ):
        if tensor.requires_grad:
            continue
        value = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
        h = hashlib.sha256()
        for start in range(0, raw.numel(), 16 * 1024**2):
            h.update(raw[start : start + 16 * 1024**2].cpu().numpy().tobytes())
        hashes[name] = h.hexdigest()
    return hashes
