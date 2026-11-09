"""Qwen3.8-Flash-Next as one frozen HF model with packed NVFP4 experts, on one GPU.

Not a learner: the RL and distillation learners build ``training.flash``, whose
experts are BF16 (training/FLASH_RL.md, "Expert storage"). This loader serves the
single-GPU replays (route statistics, prune probes) of the unpruned 512-expert
release, whose BF16 experts would not fit one GPU; the RL learner takes only the
vision tower from here.

Ported from ``experiment/ar25-flash-router-backprop`` (``flash_backprop_model.py``)
and generalised to the unpruned 512-expert checkpoint. Only the routed expert
projections are NVFP4 in the ModelOpt checkpoint; every other tensor is loaded
exactly as stored (BF16; the PLE n-gram table FP8, memory-mapped on the host
and gathered row by row). Experts are ``FrozenNVFP4Experts``: packed buffers
that are dequantised per selected expert in forward and again in backward for
dX, never optimizer parameters. LoRA attaches through PEFT to the attention
and GDN projections vLLM serves adapters on.

Transformers must be 5.16.1: it ships ``Qwen4ExpForCausalLM`` but cannot load
this checkpoint through its quantizer, hence the explicit loader.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import re
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.flash.lora_targets import DENSE_TARGETS, peft_adapter_config, validate_target_modules
from training.frozen_nvfp4 import FrozenNVFP4Expert, FrozenNVFP4Experts, FrozenNVFP4Linear

FLASH_TRANSFORMERS_VERSION = "5.16.1"
FLASH_MODEL_TYPE = "qwen4_exp"
# The release routes 10 of 512 experts; a physical prune
# (scripts/prune_flashnext_experts.py) keeps any count
# above top-k, e.g. the e256..e461 ladder or the score-aware plans.
FLASH_RELEASE_EXPERTS = 512
FLASH_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)
FLASH_SHARED_EXPERT_LORA_TARGETS = DENSE_TARGETS["mlp.shared_expert"]
# Module paths a LoRA target may live under; anything else (routed experts,
# shared experts, routers, PLE, the head) is refused at attach time.
_LORA_PARENT_PATTERN = re.compile(r"^model\.layers\.\d+\.(self_attn|linear_attn)\.[a-z_]+$")


class CheckpointReader:
    """All safetensors shards of a checkpoint, memory-mapped once."""

    def __init__(self, path: str | Path):
        from safetensors import safe_open

        self.path = Path(path)
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._stack = ExitStack()
        self._shards = {
            name: self._stack.enter_context(safe_open(self.path / name, framework="pt", device="cpu"))
            for name in sorted(set(self.weight_map.values()))
        }

    def tensor(self, key: str) -> torch.Tensor:
        return self._shards[self.weight_map[key]].get_tensor(key)

    def close(self) -> None:
        self._stack.close()


class FrozenHostShardedEmbedding(nn.Module):
    """Gather only requested FP8 PLE rows on CPU, then apply the checkpoint scale."""

    def __init__(self, shards: list[torch.Tensor], scale: torch.Tensor):
        super().__init__()
        if not shards or any(shard.ndim != 2 for shard in shards):
            raise ValueError("PLE needs nonempty two-dimensional shards")
        if any(shard.device.type != "cpu" or shard.requires_grad for shard in shards):
            raise ValueError("PLE shards must be frozen CPU tensors")
        if len({shard.shape[1] for shard in shards}) != 1 or scale.numel() != 1:
            raise ValueError("PLE shard widths or scalar scale disagree")
        # Kept out of model.to()/state_dict(): this table is immutable host data.
        # ``weight`` is a plain attribute, not a buffer: the HF PLE layer reads
        # ``ngram_embedding.weight.device`` to place its indices, and a buffer
        # would follow the model to the GPU (the trainer moves the whole model
        # after loading) and turn every lookup into a device round trip.
        self.shards = tuple(shards)
        self.scale = scale.detach().to(device="cpu", dtype=torch.float32).reshape(())
        self.weight = torch.empty(0)
        self.offsets = [0]
        for shard in shards:
            self.offsets.append(self.offsets[-1] + shard.shape[0])
        self.num_embeddings = self.offsets[-1]
        self.embedding_dim = shards[0].shape[1]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Indices may arrive on any device; the gather runs on the host and the
        # caller moves the rows back next to its activations.
        flat = indices.reshape(-1).to(device="cpu").long()
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.num_embeddings):
            raise IndexError("PLE index outside checkpoint vocabulary")
        result = torch.empty((flat.numel(), self.embedding_dim), dtype=torch.bfloat16)
        for shard, begin, end in zip(self.shards, self.offsets[:-1], self.offsets[1:], strict=True):
            positions = ((flat >= begin) & (flat < end)).nonzero(as_tuple=True)[0]
            if positions.numel():
                rows = shard.index_select(0, flat[positions] - begin).float()
                result[positions] = (rows * self.scale).to(torch.bfloat16)
        return result.reshape(*indices.shape, self.embedding_dim)


def checkpoint_to_text_name(name: str) -> str | None:
    """Map a checkpoint tensor to the text-only model, or None for vision/MTP."""
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name == "lm_head.weight":
        return name
    if name.startswith(("mtp.", "model.mtp.", "model.visual.")):
        return None
    raise ValueError(f"Unexpected checkpoint tensor outside text/vision/MTP: {name}")


_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_EXPERT_TENSOR_SUFFIXES = ("weight", "weight_scale", "weight_scale_2")


def load_expert_bank(
    reader: Any, layer_index: int, num_experts: int, device: torch.device
) -> tuple[FrozenNVFP4Experts, set[str]]:
    """One layer's routed experts as frozen packed NVFP4, in three host-to-device
    copies per projection instead of one per tensor (221k tensors for the
    unpruned checkpoint, which is where a per-tensor load spends its minutes).
    Returns the bank and the checkpoint keys it consumed."""
    consumed: set[str] = set()
    stacked: dict[tuple[str, str], torch.Tensor] = {}
    for projection in _EXPERT_PROJECTIONS:
        for suffix in _EXPERT_TENSOR_SUFFIXES:
            keys = [
                f"model.language_model.layers.{layer_index}.mlp.experts.{expert}.{projection}.{suffix}"
                for expert in range(num_experts)
            ]
            consumed.update(keys)
            # Each expert's tensor is a contiguous slice of the stacked device
            # copy; FrozenNVFP4Linear keeps that slice as its buffer.
            stacked[projection, suffix] = torch.stack([reader.tensor(key) for key in keys]).to(device)
        # Weight-only W4A16 like the serving Marlin path: activations stay BF16.
        consumed.update(
            f"model.language_model.layers.{layer_index}.mlp.experts.{expert}.{projection}.input_scale"
            for expert in range(num_experts)
        )
    experts = [
        FrozenNVFP4Expert(
            *(
                FrozenNVFP4Linear(*(stacked[projection, suffix][expert] for suffix in _EXPERT_TENSOR_SUFFIXES))
                for projection in _EXPERT_PROJECTIONS
            )
        )
        for expert in range(num_experts)
    ]
    return FrozenNVFP4Experts(experts), consumed


def flash_checkpoint_config(checkpoint: str | Path) -> dict[str, Any]:
    """The raw config.json, checked to be the Flash ModelOpt NVFP4 checkpoint."""
    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    if raw.get("model_type") != FLASH_MODEL_TYPE or raw.get("quantization_config", {}).get("quant_algo") != "NVFP4":
        raise ValueError(f"{checkpoint}: expected the Flash ModelOpt NVFP4 checkpoint")
    text = raw["text_config"]
    shape = (text["num_hidden_layers"], text["num_experts_per_tok"], text["hidden_size"])
    if shape != (48, 10, 2560) or not shape[1] < int(text["num_experts"]) <= FLASH_RELEASE_EXPERTS:
        raise ValueError(
            f"{checkpoint}: unsupported Flash shape layers/top_k/hidden={shape}, experts={text['num_experts']}"
        )
    return raw


def load_flash_text_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    progress: Callable[[str], None] = print,
    expert_backend: str = "reference",
) -> nn.Module:
    """Load the text model on ``device`` with frozen packed NVFP4 experts.

    Every parameter is frozen. ``expert_backend="reference"`` runs the
    checkpoint-native dequantising GEMMs (one packed copy of the experts on
    the GPU). Other backends are not ported yet and are refused.
    """
    if expert_backend != "reference":
        raise ValueError(f"expert_backend {expert_backend!r} is not available; only 'reference' is ported")
    import transformers
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        Qwen4ExpTextRotaryEmbedding,
    )

    if transformers.__version__ != FLASH_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Flash training is pinned to Transformers {FLASH_TRANSFORMERS_VERSION}; found {transformers.__version__}"
        )
    path = Path(checkpoint)
    raw = flash_checkpoint_config(path)
    config = Qwen4ExpTextConfig.from_dict(raw["text_config"])
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    config.use_cache = False
    config.output_router_logits = False
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    model.requires_grad_(False)
    reader = CheckpointReader(path)
    # The reader owns the mmap storage behind every host PLE lookup.
    model._flash_checkpoint_reader = reader
    device = torch.device(device)
    consumed: set[str] = set()
    for layer_index, layer in enumerate(model.model.layers):
        layer.mlp.experts, keys = load_expert_bank(reader, layer_index, config.num_experts, device)
        consumed.update(keys)
        if layer_index % 8 == 7 or layer_index == 47:
            progress(f"loaded packed frozen experts for layers up to {layer_index + 1}/48")

    ple_layer = model.model.layers[1].ple.ple_embedding
    ple_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    shards = []
    for index in range(config.split_ngram_parts):
        key = f"{ple_prefix}.shard_{index}.weight"
        shards.append(reader.tensor(key))
        consumed.add(key)
    scale_key = f"{ple_prefix}.weight_scale"
    ple_layer.ngram_embedding = FrozenHostShardedEmbedding(shards, reader.tensor(scale_key))
    consumed.add(scale_key)
    # RoPE frequencies are nonpersistent buffers: rebuild them from config.
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config).to(device)

    expected = {name: tensor for name, tensor in model.state_dict().items() if ".mlp.experts." not in name}
    parameter_names = set(dict(model.named_parameters()))
    loaded: set[str] = set()
    skipped: list[str] = []
    for source_name in reader.weight_map:
        if source_name in consumed:
            continue
        target_name = checkpoint_to_text_name(source_name)
        if target_name is None:
            skipped.append(source_name)
            continue
        if target_name not in expected:
            raise ValueError(f"Unmapped Flash checkpoint tensor: {source_name} -> {target_name}")
        value = reader.tensor(source_name)
        if tuple(value.shape) != tuple(expected[target_name].shape):
            raise ValueError(f"Shape mismatch for {source_name}: {value.shape} vs {expected[target_name].shape}")
        parent_name, leaf = target_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        value = value.to(device)
        if target_name in parameter_names:
            setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
        else:
            setattr(parent, leaf, value)
        loaded.add(target_name)
    missing = sorted(set(expected) - loaded)
    if missing:
        raise ValueError(f"Missing Flash checkpoint tensors: {missing[:20]} ({len(missing)} total)")
    meta = [
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if value.is_meta
    ]
    if meta:
        raise RuntimeError(f"Unmaterialized model tensors: {meta[:20]}")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Loader unexpectedly created trainable parameters")
    if not all(re.match(r"^(mtp\.|model\.mtp\.|model\.visual\.)", key) for key in skipped):
        raise RuntimeError("Unexpected omitted checkpoint weights")
    # A config-built model has no checkpoint identity; PEFT copies the model
    # object's name_or_path into the adapter it saves, and the runtime export
    # keys its Qwen 3.x namespace rewrite on that name.
    model.name_or_path = str(path)
    model.config._name_or_path = str(path)
    model._flash_expert_backend = expert_backend
    model.eval()
    progress(
        f"loaded Flash text weights ({config.num_experts} experts/layer); skipped {len(skipped)} vision/MTP tensors; PLE on CPU"
    )
    return model


_VISION_PREFIX = "model.visual."


def load_flash_vision_tower(
    checkpoint: str | Path, *, device: str | torch.device, reader: Any | None = None
) -> nn.Module:
    """The checkpoint's vision tower (``model.visual.*``, BF16 in the release:
    ModelOpt quantized only the routed experts), frozen and in eval mode.

    ARC-2 sends its task grids and ``show()`` output as images; the rollout
    servers encode them with this same tower, so the trainer must too."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    path = Path(checkpoint)
    raw = json.loads((path / "config.json").read_text())
    config = Qwen4ExpVisionConfig.from_dict(raw["vision_config"])
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        visual = Qwen4ExpVisionModel(config)
    reader = reader if reader is not None else CheckpointReader(path)
    keys = sorted(key for key in reader.weight_map if key.startswith(_VISION_PREFIX))
    expected = dict(visual.state_dict())
    if {key.removeprefix(_VISION_PREFIX) for key in keys} - set(expected):
        raise ValueError("checkpoint vision tensors the tower does not declare")
    device = torch.device(device)
    parameter_names = set(dict(visual.named_parameters()))
    loaded = set()
    for key in keys:
        name = key.removeprefix(_VISION_PREFIX)
        value = reader.tensor(key)
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"vision tensor shape mismatch for {key}: {tuple(value.shape)}")
        parent_name, _, leaf = name.rpartition(".")
        parent = visual.get_submodule(parent_name) if parent_name else visual
        value = value.to(device=device, dtype=torch.bfloat16)
        if name in parameter_names:
            setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
        else:
            setattr(parent, leaf, value)
        loaded.add(name)
    # Nonpersistent buffers (rotary tables) are rebuilt, not loaded.
    missing = sorted(name for name in expected if name not in loaded)
    if missing:
        raise ValueError(f"missing vision tensors: {missing[:10]} ({len(missing)} total)")
    # The rotary table is a nonpersistent buffer built at construction (on the
    # meta device here): rebuild it on the tower's device, as from_pretrained would.
    rotary = visual.rotary_pos_emb
    visual.rotary_pos_emb = type(rotary)(rotary.dim, theta=rotary.theta).to(device)
    stray = [
        name
        for name, value in list(visual.named_parameters()) + list(visual.named_buffers())
        if value.is_meta
    ]
    if stray:
        raise RuntimeError(f"vision tensors left on the meta device: {stray[:10]}")
    visual.requires_grad_(False)
    visual.eval()
    return visual


class FlashMultimodalDecoder(nn.Module):
    """``Qwen4ExpModel.forward`` over the trainer's own text model.

    Embeds the native token ids, scatters the frozen vision tower's image
    features at the image placeholder ids, and runs the text model on those
    embeddings with the native ids for its n-gram (PLE) layer and the M-RoPE
    positions the batch supplies (``training.multimodal.prepare_training_media``,
    the rollout servers' own layout). The text model is held, not registered:
    the policy's parameters, LoRA state and exports stay exactly the text
    model's."""

    def __init__(self, visual: nn.Module, text_model: nn.Module, *, image_token_id: int):
        super().__init__()
        self.visual = visual
        object.__setattr__(self, "_text_model", text_model)
        self.image_token_id = int(image_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        if mm_token_type_ids is not None and (
            mm_token_type_ids.shape != input_ids.shape
            or not torch.equal(mm_token_type_ids, (input_ids == self.image_token_id).to(mm_token_type_ids.dtype))
        ):
            raise ValueError("mm_token_type_ids must identify exactly the image placeholders")
        text = self._text_model
        if pixel_values is None and image_grid_thw is not None:
            raise ValueError("image grids without pixels")
        if pixel_values is not None and (image_grid_thw is None or position_ids is None):
            raise ValueError("image pixels need their grids and M-RoPE positions")
        # Every batch embeds through this one call, with or without images, so
        # FSDP ranks holding different batch types issue the same collectives.
        # It enters the Flash FSDP root before its sharded embedding child; HF
        # text models use the ordinary embedding accessor.
        embed = getattr(text, "embed_input_ids", None)
        inputs_embeds = text.get_input_embeddings()(input_ids) if embed is None else embed(input_ids)
        if pixel_values is not None:
            with torch.no_grad():  # frozen tower: nothing upstream of the embeddings trains
                image_embeds = self.visual(
                    pixel_values.to(dtype=torch.bfloat16), grid_thw=image_grid_thw, return_dict=True
                ).pooler_output
            mask = input_ids == self.image_token_id
            if int(mask.sum()) != int(image_embeds.shape[0]):
                raise ValueError(
                    f"image placeholders ({int(mask.sum())}) differ from image features ({image_embeds.shape[0]})"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                mask.unsqueeze(-1), image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        return text(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ple_input_ids=input_ids,
            **kwargs,
        )


def attach_flash_multimodal_decoder(model: nn.Module, checkpoint: str | Path) -> FlashMultimodalDecoder:
    """Load the vision tower beside a loaded Flash text model and attach the
    multimodal decoder to it, unregistered (``flash_multimodal_decoder``)."""
    raw = flash_checkpoint_config(checkpoint)
    device = next(model.parameters()).device
    visual = load_flash_vision_tower(
        checkpoint, device=device, reader=getattr(model, "_flash_checkpoint_reader", None)
    )
    decoder = FlashMultimodalDecoder(visual, model.model, image_token_id=int(raw["image_token_id"]))
    object.__setattr__(model, "flash_multimodal_decoder", decoder)
    return decoder


def attach_flash_lora(
    model: nn.Module,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str] | tuple[str, ...] = FLASH_LORA_TARGET_MODULES,
):
    """PEFT LoRA on the BF16 attention/GDN projections only; everything else stays frozen."""
    from peft import LoraConfig, get_peft_model

    targets = [str(name) for name in target_modules]
    unknown = sorted(set(targets) - set(FLASH_LORA_TARGET_MODULES))
    if not targets or unknown:
        raise ValueError(
            f"Flash LoRA targets must be a nonempty subset of {list(FLASH_LORA_TARGET_MODULES)}; got {targets}"
        )
    lora_config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, lora_config)
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        path = name.removeprefix("base_model.model.")
        if not _LORA_PARENT_PATTERN.match(path):
            raise RuntimeError(f"LoRA attached outside the served projections: {name}")
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attached no trainable parameters")
    return peft_model


def enable_flash_gradient_checkpointing(model: nn.Module) -> None:
    """Per-decoder-layer activation checkpointing (non-reentrant) on the base model."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def decoder_and_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def _chunk_logprobs(hidden: torch.Tensor, targets: torch.Tensor, head: nn.Module) -> torch.Tensor:
    logits = head(hidden).float()
    return torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)


def selected_token_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    head_chunk: int = 256,
) -> torch.Tensor:
    """Teacher-forced logprobs of the tokens where ``target_mask`` is true.

    Scores only selected positions and checkpoints the vocabulary head per
    chunk, so a 32k-token sequence never materialises full logits.
    Returns a [batch, tokens-1] float32 tensor, zero where unselected.
    """
    if input_ids.ndim != 2 or target_mask.shape != input_ids.shape:
        raise ValueError("input_ids and target_mask must be [batch, tokens] of equal shape")
    decoder, head = decoder_and_head(model)
    output = decoder(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[:, :-1, :]
    mask = target_mask[:, 1:].to(hidden.device)
    targets = input_ids[:, 1:].to(hidden.device)
    selected_hidden, selected_targets = hidden[mask], targets[mask]
    chunks = []
    for offset in range(0, len(selected_hidden), head_chunk):
        hidden_chunk = selected_hidden[offset : offset + head_chunk]
        target_chunk = selected_targets[offset : offset + head_chunk]
        if hidden_chunk.requires_grad and torch.is_grad_enabled():
            values = checkpoint(_chunk_logprobs, hidden_chunk, target_chunk, head, use_reentrant=False)
        else:
            values = _chunk_logprobs(hidden_chunk, target_chunk, head)
        chunks.append(values)
    if not chunks:
        raise ValueError("no target positions selected")
    result = torch.zeros(targets.shape, device=hidden.device, dtype=torch.float32)
    result[mask] = torch.cat(chunks)
    return result


def flash_lora_shapes(
    text_config: dict[str, Any],
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """(fan_in, fan_out) of every LoRA target, per layer type, from config.json.

    Attention layers carry q/k/v/o_proj (q_proj holds the query and its sigmoid
    gate side by side); GDN layers carry in_proj_qkv / in_proj_z / out_proj
    and the optional per-value-head in_proj_b / in_proj_a projections.
    These are the exact shapes PEFT records for ``attach_flash_lora``.
    """
    hidden = int(text_config["hidden_size"])
    heads = int(text_config["num_attention_heads"])
    head_dim = int(text_config["head_dim"])
    kv_heads = int(text_config["num_key_value_heads"])
    key_dim = int(text_config["linear_num_key_heads"]) * int(text_config["linear_key_head_dim"])
    value_dim = int(text_config["linear_num_value_heads"]) * int(text_config["linear_value_head_dim"])
    attention = {
        "q_proj": (hidden, heads * head_dim * 2),
        "k_proj": (hidden, kv_heads * head_dim),
        "v_proj": (hidden, kv_heads * head_dim),
        "o_proj": (heads * head_dim, hidden),
    }
    gdn = {
        "in_proj_qkv": (hidden, key_dim * 2 + value_dim),
        "in_proj_z": (hidden, value_dim),
        "out_proj": (value_dim, hidden),
        "in_proj_b": (hidden, int(text_config["linear_num_value_heads"])),
        "in_proj_a": (hidden, int(text_config["linear_num_value_heads"])),
    }
    return attention, gdn


def write_flash_peft_adapter(
    path: Path,
    *,
    checkpoint: str,
    text_config: dict[str, Any],
    rank: int,
    alpha: float,
    b_scale: float = 0.0,
    seed: int = 0,
    target_modules: list[str] | tuple[str, ...] = FLASH_LORA_TARGET_MODULES,
) -> int:
    """Write a PEFT-layout adapter over the Flash text model, as the trainer saves it.

    ``b_scale=0`` gives the identity adapter (lora_B = 0, PEFT's own init): the
    exact policy a fresh trainer publishes before its first update, without
    loading the model. A positive ``b_scale`` gives a random adapter for
    serving probes. Shared-expert gate/up/down and GDN b/a targets are opt-in;
    shared targets are emitted only under mlp.shared_expert, never routed experts.
    Returns the number of tensors written.
    """
    from safetensors.torch import save_file

    targets = list(validate_target_modules([str(module) for module in target_modules]))
    attention, gdn = flash_lora_shapes(text_config)
    shared = {}
    if set(targets) & set(FLASH_SHARED_EXPERT_LORA_TARGETS):
        hidden = int(text_config["hidden_size"])
        intermediate = int(text_config["shared_expert_intermediate_size"])
        shared = {
            "gate_proj": (hidden, intermediate),
            "up_proj": (hidden, intermediate),
            "down_proj": (intermediate, hidden),
        }
    generator = torch.Generator().manual_seed(int(seed))
    tensors: dict[str, torch.Tensor] = {}
    for index, layer_type in enumerate(text_config["layer_types"]):
        if layer_type == "full_attention":
            block, shapes = "self_attn", attention
        elif layer_type == "linear_attention":
            block, shapes = "linear_attn", gdn
        else:
            raise ValueError(f"unknown layer type {layer_type!r} at layer {index}")
        layer_shapes = {f"{block}.{name}": shape for name, shape in shapes.items()}
        layer_shapes.update({f"mlp.shared_expert.{name}": shape for name, shape in shared.items()})
        for module, (fan_in, fan_out) in layer_shapes.items():
            name = module.rsplit(".", 1)[-1]
            if name not in targets:
                continue
            prefix = f"base_model.model.model.layers.{index}.{module}"
            a = torch.randn(rank, fan_in, generator=generator) / fan_in**0.5
            b = torch.randn(fan_out, rank, generator=generator) * float(b_scale)
            tensors[f"{prefix}.lora_A.weight"] = a.to(torch.float32).contiguous()
            tensors[f"{prefix}.lora_B.weight"] = b.to(torch.float32).contiguous()
    if not tensors:
        raise ValueError("no LoRA target matched any layer of the checkpoint")
    config = peft_adapter_config(
        base_model_name_or_path=checkpoint, r=rank, alpha=alpha, target_modules=targets
    )
    path.mkdir(parents=True, exist_ok=False)
    save_file(tensors, path / "adapter_model.safetensors", metadata={"format": "pt"})
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return len(tensors)
