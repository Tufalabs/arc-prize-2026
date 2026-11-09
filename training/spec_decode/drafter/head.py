"""The MTP head as a trainable module, and the target's frozen label head.

The head is vLLM's ``Qwen4ExpMultiTokenPredictor`` (``nvidia/mtp.py``, scheme
A) rebuilt from Transformers' ``qwen4_exp`` building blocks, which ignore
``mtp.*`` but define every piece of it:

- the target's multi-stream ``[4 x 2,560]`` goes through
  ``pre_fc_norm_hidden`` (one RMSNorm over all 10,240) and ``fc_hidden``,
  shared across the four branches;
- the next token's embedding goes through ``pre_fc_norm_embedding`` and
  ``fc_embedding``, and is added to every branch;
- one ``full_attention`` decoder layer (QSA with its indexer, a 512-expert
  MoE plus shared expert, hyper-connection mixers around both), no PLE;
- the final mixer collapses the multi-stream to the single stream the LM
  head reads, and the multi-stream itself feeds the next draft step.

The layer's attention is not its Transformers forward (dense masks, one
query per position): ``drafter.attention`` supplies it, pass 0 and draft
steps alike. Embedding and LM head are the target's, frozen, as vLLM shares
them at load.

The label head is the target's own last step: its final mixer over the
pre-final-mixer stream, then the LM head -- the target has no final norm --
so one extracted tensor is both the head's input and, a position on, its
label.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from torch import nn

#: The target the head drafts for, and the drafter directory it starts from
#: (``python -m drafter.extract_base --src TARGET``). A head is only
#: valid for the target it was trained on, so a new target (e.g. another
#: expert prune) retargets the whole chain -- extraction's engine, the labels,
#: training, evaluation and export -- through ``DRAFTER_TARGET`` and
#: ``DRAFTER_BASE``; ``drafter.jobs`` carries both into the job.
DRAFTER = Path(
    os.environ.get(
        "DRAFTER_BASE", "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform-drafter"
    )
)
TARGET = Path(
    os.environ.get("DRAFTER_TARGET", "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform")
)


def _modeling() -> Any:
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    return modeling_qwen4_exp


def head_config(root: Path = DRAFTER) -> Any:
    """The drafter checkpoint's text config, shaped to its one MTP layer."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    text = json.loads((Path(root) / "config.json").read_text())["text_config"]
    config = Qwen4ExpTextConfig(**text)
    config.num_experts = int(text.get("mtp_num_experts") or text["num_experts"])
    config.layer_types = list(text["mtp"]["layer_types"])
    config.num_hidden_layers = 1
    config.ple_layer_ids = []
    config.rope_parameters = dict(
        config.rope_parameters, rope_theta=text["mtp"].get("rope_theta", config.rope_parameters["rope_theta"])
    )
    config._attn_implementation = "sdpa"
    return config


class DrafterHead(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        m = _modeling()
        self.config = config
        hidden, streams = config.hidden_size, config.hc_count
        self.pre_fc_norm_embedding = m.Qwen4ExpTextRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = m.Qwen4ExpTextRMSNorm(hidden * streams, eps=config.rms_norm_eps)
        self.fc_embedding = nn.Linear(hidden, hidden, bias=False)
        self.fc_hidden = nn.Linear(hidden, hidden, bias=False)
        self.layer = m.Qwen4ExpTextDecoderLayer(config, 0)
        self.hyper_connection_mixer = m.Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.rotary = m.Qwen4ExpTextRotaryEmbedding(config)

    def fuse(self, stream: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """The layer's input: ``stream`` ``[n, 4H]`` and the next tokens'
        embeddings ``[n, H]`` -> ``[n, 4H]`` (branch outer, hidden inner)."""
        n, hidden, streams = stream.shape[0], self.config.hidden_size, self.config.hc_count
        embeds = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        stream = self.pre_fc_norm_hidden(stream).view(n, streams, hidden)
        stream = self.fc_hidden(stream)
        return (embeds.unsqueeze(-2) + stream).flatten(-2)

    def rope(self, positions: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin ``[n, rotary]`` at ``positions`` (all three M-RoPE axes
        equal: the model's own positions are token indices)."""
        cos, sin = self.rotary(like, positions[None])
        return cos[0], sin[0]

    def block(self, x: torch.Tensor, attend: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """The decoder layer on ``x`` ``[n, 4H]`` with attention supplied:
        ``attend`` maps the attention mixer's single stream ``[n, H]`` to the
        attention output ``[n, H]``. Returns the multi-stream ``[n, 4H]``."""
        layer = self.layer
        mixed, residual, inject = layer.attn_hyper_connection(x)
        out = attend(mixed)
        x = residual + (out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
        mixed, residual, inject = layer.mlp_hyper_connection(x)
        out = layer.mlp(mixed.unsqueeze(0)).squeeze(0)
        return residual + (out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    def sample_stream(self, multi: torch.Tensor) -> torch.Tensor:
        """The single stream the LM head reads."""
        return self.hyper_connection_mixer(multi)

    def trainable(self) -> list[tuple[str, nn.Parameter]]:
        """Everything but the indexer, whose top-k passes no gradient, and
        the rotary tables (buffers)."""
        return [(name, p) for name, p in self.named_parameters() if ".indexer." not in f".{name}."]


def _checkpoint_names(head: DrafterHead) -> dict[str, str]:
    """Head parameter name -> the drafter checkpoint's tensor name."""
    names = {}
    for name, _ in head.named_parameters():
        if name.startswith("layer."):
            names[name] = "mtp.layers.0." + name.removeprefix("layer.")
        else:
            names[name] = "mtp." + name
    return names


class _Reader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.mapping = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(str(self.root / "model.safetensors"), framework="pt") as f:
                self.mapping = {name: "model.safetensors" for name in f.keys()}
        self.handles: dict[str, Any] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.mapping[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.root / shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name)


def load_head(root: Path = DRAFTER, device: str = "cuda") -> DrafterHead:
    """The drafter checkpoint's head, bf16, every tensor accounted for. A
    checkpoint with NVFP4 experts (``drafter.nvfp4``) has them dequantized,
    as Marlin serves them."""
    from drafter import nvfp4

    config = head_config(root)
    with torch.device("meta"):
        head = DrafterHead(config)
    reader = _Reader(root)
    names = _checkpoint_names(head)
    experts: dict[str, torch.Tensor] = {}
    quantized: set[str] = set()
    if nvfp4.is_quantized(reader.mapping):
        experts = nvfp4.read_packed(reader.get, config.num_experts)
        quantized = nvfp4.expert_tensor_names(config.num_experts)
    state = {}
    for name, source in names.items():
        tensor = experts[source] if source in experts else reader.get(source)
        expected = dict(head.named_parameters())[name].shape
        if tuple(tensor.shape) != tuple(expected):
            raise ValueError(f"{source}: {tuple(tensor.shape)} against {tuple(expected)}")
        state[name] = tensor.to(torch.bfloat16)
    unused = {n for n in reader.mapping if n.startswith("mtp.")} - set(names.values()) - quantized
    # vLLM folds block_inject_weight into the final mixer's down projection
    # only when the mixer combines; the final mixer here does not.
    unused -= {"mtp.hyper_connection_mixer.block_inject_weight.weight"}
    if unused:
        raise ValueError(f"drafter tensors the head does not load: {sorted(unused)}")
    head.load_state_dict(state, strict=False, assign=True)
    head.rotary = _modeling().Qwen4ExpTextRotaryEmbedding(config)
    return head.to(device)


class LabelHead(nn.Module):
    """The target's final mixer and LM head, frozen; plus its token embedding,
    which the drafter shares."""

    def __init__(self, root: Path = TARGET, device: str = "cuda") -> None:
        super().__init__()
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

        config = Qwen4ExpTextConfig(**json.loads((Path(root) / "config.json").read_text())["text_config"])
        reader = _Reader(root)
        mixer = _modeling().Qwen4ExpTextGatedResidual(config, use_combine=False).to(torch.bfloat16)
        prefix = "model.language_model.hyper_connection_mixer."
        mixer.load_state_dict({n: reader.get(prefix + n) for n, _ in mixer.named_parameters()})
        if any(p.dtype != torch.bfloat16 for p in mixer.parameters()):
            raise ValueError("the target's final mixer is not bf16")
        self.mixer = mixer.to(device)
        self.lm_head = reader.get("lm_head.weight").to(device=device, dtype=torch.bfloat16)
        self.embed = reader.get("model.language_model.embed_tokens.weight").to(device=device, dtype=torch.bfloat16)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def single(self, stream: torch.Tensor) -> torch.Tensor:
        """The single stream the target's LM head reads, from its
        pre-final-mixer multi-stream."""
        return self.mixer(stream)


def load_trained(head: DrafterHead, path: Path) -> None:
    """A trained head's tensors (``export_state`` names) into ``head``."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    names = _checkpoint_names(head)
    if set(tensors) != set(names.values()):
        raise ValueError(f"{path}: tensors do not match the head's")
    with torch.no_grad():
        for name, p in head.named_parameters():
            p.copy_(tensors[names[name]])


def export_state(head: DrafterHead) -> dict[str, torch.Tensor]:
    """The head's tensors under the drafter checkpoint's names, bf16."""
    names = _checkpoint_names(head)
    return {names[n]: p.detach().to(torch.bfloat16).contiguous().cpu() for n, p in head.named_parameters()}


def clone_config(config: Any) -> Any:
    return copy.deepcopy(config)
