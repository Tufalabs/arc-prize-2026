"""Explicit Qwen4-Exp adapters installed while the model is constructed."""

from __future__ import annotations

import contextvars

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from .batched_qsa import BatchedQSAIndexer
from .convolution import compiled_ple_convolution, fla_convolution
from .decision_replay import expert_topk, reject_replay
from .flex_qsa import QUERY_TILE, FlexQSAAttention
from .gated_pointwise import TritonGatedNorm, TritonProjectionNorm, TritonSharedMLP, pack_qkv
from .hyper_connection import (
    ManualGatedResidual, TritonGatedResidual, residual_inject, residual_inject_norm,
)
from .input_layout import pack_padded_batch, positions_reset
from .options import FlashOptions
from .pointwise_optimizations import FlashExperts, FrozenGroupedRMSNorm

_active_lengths = contextvars.ContextVar("flash_packed_document_lengths", default=None)


def _packed_boundaries(lengths, cu_seqlens, tokens, device):
    if lengths is None:
        return cu_seqlens
    lengths = tuple(lengths)
    if any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != tokens:
        raise ValueError(f"GDN packed boundaries: packed_lengths={lengths!r} must be positive and sum to tokens={tokens}")
    if cu_seqlens is not None:
        if (
            not hasattr(cu_seqlens, "_flash_packed_lengths")
            or cu_seqlens._flash_packed_lengths != lengths
            or cu_seqlens._flash_packed_version != cu_seqlens._version
            or cu_seqlens.device != device
        ):
            raise ValueError(f"GDN packed boundaries: callers may not supply both packed_lengths={lengths!r} and cu_seq_lens_q; pass lengths alone")
        return cu_seqlens
    ends = [0]
    for length in lengths:
        ends.append(ends[-1] + length)
    with torch.inference_mode(False):
        cu_seqlens = torch.tensor(ends, dtype=torch.int32, device=device)
    cu_seqlens._flash_packed_lengths = lengths
    cu_seqlens._flash_packed_version = cu_seqlens._version
    return cu_seqlens


def validate_transformers_contract():
    """Fail with an actionable error when the required public surface changes."""

    required = {
        "Qwen4ExpTextModel": ("layers", "rotary_emb"),
        "Qwen4ExpTextDecoderLayer": ("mlp",),
        "Qwen4ExpTextGatedDeltaNet": ("in_proj_qkv", "conv1d", "norm"),
        "Qwen4ExpTextQSAIndexer": ("index_qk_proj",),
        "Qwen4ExpTextExperts": ("gate_up_proj", "down_proj"),
    }
    missing = [name for name in required if not hasattr(qwen, name)]
    if missing:
        raise RuntimeError(f"Unsupported Transformers Qwen4-Exp surface: {missing}")
    if not hasattr(qwen, "apply_rotary_pos_emb") or not hasattr(
        qwen, "Qwen4ExpTextConfig"
    ):
        raise RuntimeError("Transformers Qwen4-Exp rotary/config API is unavailable")
    return True


class FP32TopKRouter(qwen.Qwen4ExpTextTopKRouter):
    """FP32 master weights with reference BF16 routing arithmetic."""

    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        return router_forward(self, hidden_states, self.layer_idx)


class StockTopKRouter(qwen.Qwen4ExpTextTopKRouter):
    """Native HF routing; it never consults a decision tape, so refuse one."""

    def forward(self, hidden_states):
        reject_replay("router_backend='stock'", "use router_backend='fp32_reference'")
        return super().forward(hidden_states)


def router_forward(self, hidden_states, layer_idx=None):
    x = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(x, self.weight.to(x.dtype))
    # Optional probe hook: set only by router probes, absent on ordinary routers.
    excluded = getattr(self, "probe_excluded_experts", None)
    if excluded is not None:
        logits = logits.masked_fill(excluded, float("-inf"))
    probs = logits.float().softmax(-1)
    values, indices = expert_topk(
        probs, self.top_k, layer_idx, _active_lengths.get()
    )
    if self.norm_topk_prob:
        values = values / values.sum(-1, keepdim=True)
    # Optional probe hook, as above.
    recorder = getattr(self, "route_recorder", None)
    if recorder is not None:
        recorder.record(self.route_layer, indices, values)
    return logits, values.to(logits.dtype), indices


class PackedNGramEmbedding(qwen.Qwen4ExpTextNGramEmbedding):
    """PLE n-gram hashing that restarts history at each packed document."""

    def forward(self, input_ids, past_key_values):
        lengths = _active_lengths.get()
        if lengths is None:
            return super().forward(input_ids, past_key_values)
        if past_key_values is not None or sum(lengths) != input_ids.shape[1]:
            raise ValueError(
                "Packed PLE n-gram embedding requires complete cache-free documents"
            )
        # Native hashing only resets history at EOS. Explicit packed documents
        # need not end in EOS, so each one starts from fresh EOS context.
        return torch.cat(
            [
                super(PackedNGramEmbedding, self).forward(piece, None)
                for piece in input_ids.split(lengths, dim=1)
            ],
            dim=1,
        )


class PackedPLELayer(qwen.Qwen4ExpTextPLELayer):
    """PLE embedding and convolution that respect packed-document boundaries."""

    def __init__(self, config, layer_idx, ple_layer_index, *, convolution="native"):
        super().__init__(config, layer_idx, ple_layer_index)
        self.convolution = convolution
        self.ple_embedding = PackedNGramEmbedding(
            config, config.ple_embed_dim, layer_idx, ple_layer_index
        )

    def forward(self, hidden_states, input_ids, past_key_values, conv_mask=None):
        # CUDA autocast promotes HF's gate reduction to FP32, which promotes
        # the residual stream too. Keep the native BF16 PLE arithmetic used
        # outside autocast; the fused decoder requires low-precision states.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return super().forward(hidden_states, input_ids, past_key_values, conv_mask=conv_mask)

    def _short_conv(self, hidden_states, past_key_values):
        lengths = _active_lengths.get()
        if self.convolution == "compiled":
            if past_key_values is not None:
                raise ValueError("Compiled PLE convolution requires cache-free inputs")
            positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            if lengths is not None:
                ends = torch.tensor(lengths, device=hidden_states.device).cumsum(0)
                starts = torch.cat((ends.new_zeros(1), ends[:-1]))
                positions = positions - starts[torch.bucketize(positions, ends, right=True)]
            return compiled_ple_convolution(
                hidden_states, self.conv1d.weight, self.conv1d.dilation[0], positions
            )
        if lengths is None:
            return super()._short_conv(hidden_states, past_key_values)
        if past_key_values is not None or sum(lengths) != hidden_states.shape[1]:
            raise ValueError(
                "Packed PLE convolution requires complete cache-free documents"
            )
        return torch.cat(
            [
                super(PackedPLELayer, self)._short_conv(piece, None)
                for piece in hidden_states.split(lengths, dim=1)
            ],
            dim=1,
        )


class QualifiedGatedDeltaNet(qwen.Qwen4ExpTextGatedDeltaNet):
    """GDN with explicitly injected kernels and packed-document handling."""

    def __init__(self, config, layer_idx, *, chunk_rule, causal_conv, options=FlashOptions()):
        super().__init__(config, layer_idx)
        self._flash_chunk_rule = chunk_rule
        self._flash_causal_conv = causal_conv
        self.options = options
        self.reference_gates = False
        self.pack_qkv = None
        if options.fuse_gdn_qkv:
            if options.gdn_grouped_heads or self.head_k_dim != self.head_v_dim:
                raise ValueError("Fused GDN QKV packing requires repeated heads with equal head dimensions")
            self.pack_qkv = pack_qkv
        if options.fuse_gdn_norm:
            self.norm = TritonGatedNorm(
                self.head_v_dim, eps=config.rms_norm_eps,
                activation=self.norm.activation, serving=options.pointwise_rounding == "vllm",
            )

    def _convolution(self, mixed_qkv, packed_lengths, cu_seqlens=None, **kwargs):
        if self.options.gdn_convolution == "fla":
            cu_seqlens = _packed_boundaries(
                packed_lengths, cu_seqlens, mixed_qkv.shape[-1], mixed_qkv.device
            )
            return fla_convolution(
                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                self.activation, cu_seqlens,
            )
        def call(piece):
            return self._flash_causal_conv(
                piece,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

        if packed_lengths is None:
            return call(mixed_qkv)
        lengths = tuple(int(value) for value in packed_lengths)
        if sum(lengths) != mixed_qkv.shape[-1] or mixed_qkv.shape[0] != 1:
            raise ValueError("Invalid packed GDN convolution layout")
        return torch.cat(
            [call(piece) for piece in mixed_qkv.split(lengths, dim=-1)], dim=-1
        )

    @torch.compiler.disable
    def _reference_gates(self, b, a):
        return b.sigmoid(), -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    def forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        **kwargs,
    ):
        packed_lengths = kwargs.pop("packed_lengths", None)
        cu_seqlens = kwargs.pop("cu_seq_lens_q", None)
        if packed_lengths is not None:
            if hidden_states.shape[0] != 1:
                raise ValueError(f"GDN packed boundaries: batch={hidden_states.shape[0]}; requires batch=1")
            cu_seqlens = _packed_boundaries(
                packed_lengths, cu_seqlens, hidden_states.shape[1], hidden_states.device
            )
        elif cu_seqlens is not None and self.options.gdn_convolution == "native":
            raise ValueError("GDN native convolution: cu_seq_lens_q requires packed_lengths")
        hidden_states = qwen.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state(self.layer_idx, state_idx=0)
        )
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)

        if (
            use_precomputed_states
            and seq_len == 1
            and not cache_params.layers[self.layer_idx].record_past
        ):
            mixed_qkv = qwen.causal_conv1d_update(
                mixed_qkv,
                cache_params.layers[self.layer_idx].conv_states[0],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv,
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )
            mixed_qkv = self._convolution(
                mixed_qkv, packed_lengths, cu_seqlens=cu_seqlens, **kwargs
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        fused_qkv = self.options.fuse_gdn_qkv and mixed_qkv.is_cuda
        if fused_qkv:
            query, key, value = self.pack_qkv(
                mixed_qkv.transpose(1, 2), self.num_k_heads, self.num_v_heads, self.head_k_dim
            )
        else:
            query, key, value = torch.split(
                mixed_qkv.transpose(1, 2),
                [self.key_dim, self.key_dim, self.value_dim],
                dim=-1,
            )
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if self.reference_gates:
            beta, g = self._reference_gates(b, a)
        else:
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        grouped_heads = self.options.gdn_grouped_heads and not (
            use_precomputed_states and seq_len == 1
        )
        if self.num_v_heads // self.num_k_heads > 1 and not (grouped_heads or fused_qkv):
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        recurrent_state = (
            cache_params.layers[self.layer_idx].recurrent_states[0]
            if use_precomputed_states
            else None
        )
        common = dict(
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        if use_precomputed_states and seq_len == 1:
            core, last_state = qwen.torch_recurrent_gated_delta_rule(
                query, key, value, **common
            )
        else:
            core, last_state = self._flash_chunk_rule(query, key, value, **common)
        if cache_params is not None:
            cache_params.update_recurrent_state(last_state, self.layer_idx)
        core = core.reshape(-1, self.head_v_dim)
        core = self.norm(core, z.reshape(-1, self.head_v_dim))
        core = core.reshape(batch_size, seq_len, -1)
        return self.out_proj(core)


def _replace_grouped_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, qwen.Qwen4ExpTextRMSNorm) and child.group_size is not None:
            replacement = FrozenGroupedRMSNorm(
                child.weight.shape[0], group_size=child.group_size, eps=child.eps
            )
            setattr(module, name, replacement)
        else:
            _replace_grouped_norms(child)


def _gated_residual(config, options, **kwargs):
    """The hyper-connection residual module that options.hyper_connection selects."""
    residual = ManualGatedResidual if options.hyper_connection == "triton_manual" else TritonGatedResidual
    return residual(config, serving=options.pointwise_rounding == "vllm", **kwargs)


class FlashDecoderLayer(qwen.Qwen4ExpTextDecoderLayer):
    def __init__(
        self,
        config,
        layer_idx,
        *,
        flex_qsa,
        fp32_router,
        chunk_rule,
        causal_conv,
        fuse_norm_tail,
        options=FlashOptions(),
    ):
        super().__init__(config, layer_idx)
        self.options = options
        self.compiled_forward = None
        if self.layer_type == "linear_attention":
            self.linear_attn = QualifiedGatedDeltaNet(
                config,
                layer_idx,
                chunk_rule=chunk_rule,
                causal_conv=causal_conv,
                options=options,
            )
        elif flex_qsa:
            self.self_attn = FlexQSAAttention(
                config, layer_idx, qsa_kernel=options.qsa_kernel,
                selector_policy=options.selector_policy,
            )
        elif hasattr(self.self_attn, "indexer"):
            # SDPA remains the teacher/default attention backend, but it must
            # use the same qualified selector as FlexQSA. The upstream QSA
            # indexer performs a Python loop and host sync for every query.
            self.self_attn.indexer = BatchedQSAIndexer(
                config, layer_idx, selector_policy=options.selector_policy
            )
        self.mlp.gate = FP32TopKRouter(config, layer_idx) if fp32_router else StockTopKRouter(config)
        if self.ple is not None:
            ple_index = config.ple_layer_ids.index(layer_idx + 1)
            self.ple = PackedPLELayer(config, layer_idx, ple_index, convolution=options.ple_convolution)
        self.mlp.experts = FlashExperts(
            config, optimization=options.expert_optimization, fuse_gate=options.fuse_mlp_gates,
        )
        if options.hyper_connection != "none":
            self.attn_hyper_connection = _gated_residual(config, options)
            self.mlp_hyper_connection = _gated_residual(config, options)
        if options.fuse_qsa_norm and hasattr(self, "self_attn"):
            self.self_attn.q_norm = TritonProjectionNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)
            self.self_attn.k_norm = TritonProjectionNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)
        if options.fuse_mlp_gates:
            if config.hidden_act not in ("silu", "swish"):
                raise ValueError("Fused MLP gates require SiLU")
            self.mlp.shared_expert = TritonSharedMLP(config, config.shared_expert_intermediate_size)
        if fuse_norm_tail:
            _replace_grouped_norms(self)

    def _eager_forward(self):
        """The uncompiled decoder forward: the hyper-connection path when enabled."""
        return self._hyper_forward if self.options.hyper_connection != "none" else super().forward

    def forward(self, *args, **kwargs):
        call = self.compiled_forward if self.compiled_forward is not None else self._eager_forward()
        lengths = kwargs.get("packed_lengths")
        if lengths is None:
            return call(*args, **kwargs)
        token = _active_lengths.set(tuple(int(value) for value in lengths))
        try:
            return call(*args, **kwargs)
        finally:
            _active_lengths.reset(token)

    def _hyper_forward(
        self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None,
        past_key_values=None, ple_input_ids=None, **kwargs,
    ):
        if not hidden_states.is_cuda:
            return super().forward(
                hidden_states, position_embeddings, attention_mask=attention_mask,
                conv_mask=conv_mask, past_key_values=past_key_values,
                ple_input_ids=ple_input_ids, **kwargs,
            )
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask
            )
        hidden_states, hyper_input, weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states, position_embeddings, attention_mask=attention_mask,
                past_key_values=past_key_values, **kwargs,
            )
        serving = self.options.pointwise_rounding == "vllm"
        if self.options.hyper_connection == "triton_manual":
            hidden_states, hyper_input, weights = self.mlp_hyper_connection(
                hyper_input, injection=(hidden_states, weights)
            )
        elif self.options.hyper_connection == "triton_inject_norm":
            norm = self.mlp_hyper_connection.hc_norm
            hidden_states, normed = residual_inject_norm(
                hyper_input, hidden_states, weights, norm.weight, norm.eps, serving
            )
            hidden_states, hyper_input, weights = self.mlp_hyper_connection(hidden_states, normed=normed)
        else:
            hidden_states = residual_inject(hyper_input, hidden_states, weights, serving)
            hidden_states, hyper_input, weights = self.mlp_hyper_connection(hidden_states)
        return residual_inject(hyper_input, self.mlp(hidden_states), weights, serving)


class FlashTextModel(qwen.Qwen4ExpTextModel):
    """Qwen4-Exp model whose Flash adapters are part of its module topology."""

    @classmethod
    def _can_set_experts_implementation(cls):
        # The upstream heuristic scans the subclass module for its decorator.
        # Our layers still contain the same decorated native Experts class.
        return True

    def __init__(
        self,
        config,
        *,
        flex_qsa,
        fp32_router,
        chunk_rule,
        causal_conv,
        fuse_norm_tail,
        options=FlashOptions(),
    ):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                FlashDecoderLayer(
                    config,
                    index,
                    flex_qsa=flex_qsa,
                    fp32_router=fp32_router,
                    chunk_rule=chunk_rule,
                    causal_conv=causal_conv,
                    fuse_norm_tail=fuse_norm_tail,
                    options=options,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        if options.hyper_connection != "none":
            self.hyper_connection_mixer = _gated_residual(config, options, use_combine=False)
        self._flash_flex_qsa = flex_qsa

    def embed_input_ids(self, input_ids):
        """Embedding forward registered at the FSDP root for image scattering."""
        return self.embed_tokens(input_ids)

    @staticmethod
    def _unpack(output, layout):
        if layout is None:
            return output
        if isinstance(output, tuple):
            return (layout.unpack(output[0]), *output[1:])
        output.last_hidden_state = layout.unpack(output.last_hidden_state)
        if getattr(output, "hidden_states", None) is not None:
            output.hidden_states = tuple(
                layout.unpack(value) for value in output.hidden_states
            )
        return output

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if not self._flash_flex_qsa:
            packed_lengths = kwargs.get("packed_lengths")
            tokens = input_ids if input_ids is not None else kwargs.get("inputs_embeds")
            if packed_lengths is not None and tokens is not None:
                # Derive GDN boundaries once per forward; every GDN layer and
                # checkpoint recompute then reuses the trusted tensor.
                kwargs["cu_seq_lens_q"] = _packed_boundaries(
                    tuple(int(value) for value in packed_lengths),
                    kwargs.get("cu_seq_lens_q"), tokens.shape[1], tokens.device,
                )
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        packed_lengths = kwargs.get("packed_lengths")
        inputs_embeds = kwargs.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        raw_ids = input_ids if input_ids is not None else kwargs.get("ple_input_ids")
        if inputs_embeds is not None:
            if inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must have shape [B, T, H]")
            if raw_ids is None:
                if self.config.ple_layer_ids:
                    raise ValueError("inputs_embeds with PLE requires raw ple_input_ids")
                raw_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            if raw_ids.shape != inputs_embeds.shape[:2] or raw_ids.device != inputs_embeds.device:
                raise ValueError("ple_input_ids must match inputs_embeds")
        if (
            raw_ids.ndim != 2
            or kwargs.get("past_key_values") is not None
            or kwargs.get("use_cache", False)
        ):
            raise ValueError(
                "flex_qsa model requires complete causal token sequences without cache"
            )
        layout = None
        positions = kwargs.get("position_ids")
        if positions is not None and (
            positions.shape not in (raw_ids.shape, (3, *raw_ids.shape))
            or positions.device != raw_ids.device
        ):
            raise ValueError("position_ids must have shape [B, T] or [3, B, T] and match tokens")
        if raw_ids.shape[0] != 1 or attention_mask is not None:
            if packed_lengths is not None or kwargs.get("cu_seq_lens_q") is not None:
                raise ValueError(
                    "Do not combine a padded public batch with explicit packed metadata"
                )
            if kwargs.get("return_dict") is False:
                raise ValueError("Padded public batches require return_dict=True")
            pad = getattr(self.config, "pad_token_id", None)
            eos = getattr(self.config, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                eos = eos[0] if eos else None
            layout = pack_padded_batch(
                raw_ids,
                attention_mask,
                kwargs.pop("position_ids", None),
                pad_token_id=0 if pad is None else pad,
                eos_token_id=0 if eos is None else eos,
                alignment=QUERY_TILE,
            )
            if inputs_embeds is None:
                input_ids = layout.input_ids
            else:
                separator = self.embed_tokens(raw_ids.new_tensor(0 if eos is None else eos))
                kwargs["inputs_embeds"] = layout.pack(inputs_embeds, separator)
            if kwargs.get("ple_input_ids") is not None:
                kwargs["ple_input_ids"] = layout.pack(kwargs["ple_input_ids"], 0 if eos is None else eos)
            raw_ids = layout.input_ids
            kwargs["position_ids"] = layout.position_ids
            kwargs["packed_lengths"] = layout.packed_lengths
            kwargs.pop("cu_seq_lens_q", None)
            packed_lengths = layout.packed_lengths
        elif positions is not None and packed_lengths is None:
            # One unpadded sequence with explicit positions: a reset would be a
            # document boundary that attention, GDN and PLE never learn about.
            if positions.ndim == 2:
                raise ValueError("Explicit position_ids require packed document metadata")
            if positions_reset(positions, [(0, 0, raw_ids.shape[1])]):
                raise ValueError("M-RoPE position_ids reset without packed document metadata")
        if packed_lengths is not None:
            packed_lengths = tuple(int(value) for value in packed_lengths)
            if sum(packed_lengths) != raw_ids.shape[1] or any(
                value <= 0 or value % QUERY_TILE for value in packed_lengths
            ):
                raise ValueError("Invalid packed document lengths")
            if positions is not None and layout is None:
                starts = [sum(packed_lengths[:index]) for index in range(len(packed_lengths))]
                if positions_reset(kwargs["position_ids"], [(0, start, start + length)
                                                            for start, length in zip(starts, packed_lengths)]):
                    raise ValueError("position_ids reset inside a packed document")
            if kwargs.get("position_ids") is None:
                kwargs["position_ids"] = torch.cat(
                    [
                        torch.arange(value, device=raw_ids.device)
                        for value in packed_lengths
                    ]
                )[None]
            kwargs["cu_seq_lens_q"] = _packed_boundaries(
                packed_lengths, kwargs.get("cu_seq_lens_q"), raw_ids.shape[1], raw_ids.device
            )
        output = super().forward(
            input_ids=input_ids,
            attention_mask={"full_attention": None, "linear_attention": None},
            **kwargs,
        )
        return self._unpack(output, layout)
