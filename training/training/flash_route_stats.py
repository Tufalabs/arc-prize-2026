"""Per-rollout expert routing statistics from token-exact Flash traces, for pruning.

REAP (arXiv 2510.13999) scores expert ``j`` of a layer by the mean, over the
tokens routed to it, of ``g_j(x) * ||f_j(x)||``: its router gate value times the
norm of its (unweighted) output. Replaying each rollout's traces through the
unpruned model records, per rollout, per layer and expert, the three sums a
REAP-style score is made of (``sum_gnorm``, ``sum_g``, ``count``). Kept per
rollout, they let any rollout weighting -- by its ARC score, relative to the
other attempts at the same game -- be applied afterwards without replaying.

A trace's model calls form chains: a call whose prompt extends the previous
call's prompt plus reply continues the chain, so the chain's last prompt plus
reply holds every token of it. Each chain replays from an empty cache in
chunks (``replay_chain``): the fast QSA indexer accepts the cache
(``training.flash_qsa_train``), so contexts far past what one forward's dense
attention mask fits cost only their KV cache.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def iter_trace_calls_with_media(path: str | Path) -> Iterator[tuple[list[int], list[int], list[str]]]:
    """(prompt ids, reply ids, image paths) of every model call in a taaf
    trace, in order. The paths (absolute, in placeholder order) are every
    image the call's prompt carries."""
    root = Path(path).parent
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt, reply = record.get("input_token_ids"), record.get("output_token_ids")
            if not isinstance(prompt, list) or not isinstance(reply, list) or not prompt:
                raise ValueError(f"{path}: a trace call lacks its token ids")
            images = [str(root / item["path"]) for item in record.get("media") or [] if item.get("path")]
            yield prompt, reply, images


def trace_chains(path: str | Path) -> list[tuple[list[int], list[tuple[int, int]], list[str]]]:
    """A trace's chains as a replay needs them. A call whose prompt extends
    the previous prompt plus reply continues the chain; anything else (a fresh
    render after a summary) starts one. Per chain: its ids (the last prompt
    plus reply), the ``[start, end)`` of every reply in it (the tokens the
    rollout's policy sampled), and its images' paths in placeholder order (its
    last call's, which carry every image it holds)."""
    result: list[tuple[list[int], list[tuple[int, int]], list[str]]] = []
    current: list[int] | None = None
    for prompt, reply, images in iter_trace_calls_with_media(path):
        if not (current is not None and len(prompt) >= len(current) and prompt[: len(current)] == current):
            result.append(([], [], []))
        current = prompt + reply
        result[-1] = (current, [*result[-1][1], (len(prompt), len(current))], images)
    return result


def truncate_chain(
    ids: list[int], replies: list[tuple[int, int]], images: list[str], limit: int, image_token_id: int
) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    """A chain cut to at most its first ``limit`` tokens, with the replies
    (or their parts) and the images that fall inside. A cut that would land
    in an image's placeholders moves back to before that image, so the chain
    keeps only whole images."""
    limit = min(limit, len(ids))
    if 0 < limit < len(ids) and ids[limit - 1] == image_token_id and ids[limit] == image_token_id:
        while limit and ids[limit - 1] == image_token_id:
            limit -= 1
    kept = ids[:limit]
    runs = sum(1 for index, token in enumerate(kept) if token == image_token_id and (index == 0 or kept[index - 1] != token))
    return kept, [(start, min(end, limit)) for start, end in replies if start < limit], images[:runs]


def chains_with_logprobs(path: str | Path) -> list[tuple[list[int], dict[int, float]]]:
    """A taaf trace's chains with, per chain, the serving engine's
    log-probability of every sampled reply token, by chain position."""
    result: list[tuple[list[int], dict[int, float]]] = []
    current: list[int] | None = None
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt, reply = record.get("input_token_ids"), record.get("output_token_ids")
            if not isinstance(prompt, list) or not isinstance(reply, list) or not prompt:
                raise ValueError(f"{path}: a trace call lacks its token ids")
            if not (current is not None and len(prompt) >= len(current) and prompt[: len(current)] == current):
                result.append(([], {}))
            current = prompt + reply
            targets = result[-1][1]
            logprobs = record.get("logprobs") or []
            if len(logprobs) == len(reply):
                for offset, item in enumerate(logprobs):
                    targets[len(prompt) + offset] = float(item["logprob"])
            result[-1] = (current, targets)
    return result


def aggregate_agreement(first: dict[str, Any], second: dict[str, Any], *, keep: int) -> dict[str, float]:
    """How alike two replays of the same tokens are where pruning looks: per
    layer, the correlation of routed-token counts and the overlap of the
    ``keep`` experts with the most gate * output norm."""
    import numpy as np

    correlations, overlaps = [], []
    for layer in range(first["count"].shape[0]):
        correlations.append(float(np.corrcoef(first["count"][layer], second["count"][layer])[0, 1]))
        top_first = set(np.argsort(-first["sum_gnorm"][layer], kind="stable")[:keep].tolist())
        top_second = set(np.argsort(-second["sum_gnorm"][layer], kind="stable")[:keep].tolist())
        overlaps.append(len(top_first & top_second) / keep)
    return {
        "min_count_correlation": min(correlations),
        "mean_count_correlation": sum(correlations) / len(correlations),
        "min_keep_overlap": min(overlaps),
        "mean_keep_overlap": sum(overlaps) / len(overlaps),
    }


@dataclass
class RouteStats:
    """Per layer and expert: sum of gate * output norm, sum of gate, routed tokens."""

    sum_gnorm: torch.Tensor
    sum_g: torch.Tensor
    count: torch.Tensor
    tokens: int = 0

    @classmethod
    def zeros(cls, layers: int, experts: int, device: torch.device | str) -> "RouteStats":
        shape = (layers, experts)
        return cls(*(torch.zeros(shape, dtype=torch.float64, device=device) for _ in range(3)))

    def reset(self) -> None:
        for tensor in (self.sum_gnorm, self.sum_g, self.count):
            tensor.zero_()
        self.tokens = 0

    def arrays(self) -> dict[str, Any]:
        return {
            "sum_gnorm": self.sum_gnorm.cpu().numpy(),
            "sum_g": self.sum_g.cpu().numpy(),
            "count": self.count.cpu().numpy(),
            "tokens": self.tokens,
        }


def install_route_recorder(model: nn.Module, stats: RouteStats) -> int:
    """Record routing into ``stats`` from every layer's frozen expert bank.

    The bank already runs each selected expert on its own tokens; the recorder
    adds, per expert call, the norm of that unweighted output and the gate
    values that weight it. The layer's output is unchanged."""
    from training.frozen_nvfp4 import FrozenNVFP4Experts

    installed = 0
    for layer_index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        if not isinstance(experts, FrozenNVFP4Experts):
            raise TypeError(f"layer {layer_index}: expected frozen NVFP4 experts, got {type(experts).__name__}")

        def recording_forward(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            _experts: FrozenNVFP4Experts = experts,
            _layer: int = layer_index,
        ) -> torch.Tensor:
            accumulator = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
            for expert_id in top_k_index.unique(sorted=True).tolist():
                token_index, slot = torch.where(top_k_index == expert_id)
                expert_output = _experts[expert_id](hidden_states[token_index])
                gate = top_k_weights[token_index, slot]
                norm = torch.linalg.vector_norm(expert_output.float(), dim=-1)
                stats.sum_gnorm[_layer, expert_id] += (gate.double() * norm.double()).sum()
                stats.sum_g[_layer, expert_id] += gate.double().sum()
                stats.count[_layer, expert_id] += token_index.numel()
                accumulator.index_add_(0, token_index, (expert_output * gate[:, None]).to(torch.float32))
            return accumulator.to(hidden_states.dtype)

        experts.forward = recording_forward
        installed += 1
    return installed


def refuse_images_as_text(ids: list[int], image_token_id: int, what: str) -> None:
    """A text replay feeds an image placeholder its own embedding where the
    served model read image features, and computes every later token without
    the image: statistics of a model that was not served. A chain that carries
    images replays through the vision tower (``replay_multimodal_chain``)."""
    placeholders = ids.count(image_token_id)
    if placeholders:
        raise ValueError(f"{what}: {placeholders:,} image placeholder tokens and no images to put in their place")


@torch.no_grad()
def replay_chain(
    model: nn.Module,
    ids: list[int],
    *,
    chunk_tokens: int,
    stats: RouteStats | None = None,
    on_chunk=None,
):
    """Prefill one chain in chunks from an empty cache; ``on_chunk(start,
    hidden)`` sees each chunk's last hidden states. Returns the last chunk's."""
    from transformers import DynamicCache

    device = next(model.parameters()).device
    cache = DynamicCache(config=model.config)
    hidden = None
    for start in range(0, len(ids), chunk_tokens):
        piece = torch.tensor([ids[start : start + chunk_tokens]], dtype=torch.long, device=device)
        hidden = model.model(input_ids=piece, past_key_values=cache, use_cache=True).last_hidden_state
        if on_chunk is not None:
            on_chunk(start, hidden)
    if stats is not None:
        stats.tokens += len(ids)
    return hidden


@torch.no_grad()
def replay_multimodal_chain(
    model: nn.Module,
    decoder: nn.Module,
    ids: list[int],
    images: list[Any],
    *,
    processor: Any,
    processing: dict[str, Any],
    chunk_tokens: int,
    stats: RouteStats | None = None,
    on_chunk=None,
):
    """``replay_chain`` for a chain whose prompt carries images: the images are
    processed as the serving stack does (``ullm.media``), encoded once by the
    frozen tower (``decoder.visual``), scattered at their placeholders chunk by
    chunk, with the chain's M-RoPE positions and native ids for the PLE layer."""
    from transformers import DynamicCache

    from training.ullm_media import ensure_ullm_media

    ensure_ullm_media()
    from ullm.media import image_layout, position_inputs, process_images

    text = model.model
    device = next(model.parameters()).device
    tensors = process_images(processor, images, processing)
    layout = image_layout(ids, tensors["image_grid_thw"], processing)
    if len(layout) != len(images):
        raise ValueError(f"{len(layout)} image placeholder runs for {len(images)} images")
    positions = position_inputs(len(ids), layout, processing)["position_ids"]
    features = decoder.visual(
        tensors["pixel_values"].to(device=device, dtype=torch.bfloat16),
        grid_thw=tensors["image_grid_thw"].to(device),
        return_dict=True,
    ).pooler_output
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    placeholders = ids_tensor == decoder.image_token_id
    if int(placeholders.sum()) != int(features.shape[0]):
        raise ValueError("image placeholders differ from encoded image features")
    feature_index = placeholders.long().cumsum(0) - 1
    cache = DynamicCache(config=model.config)
    hidden = None
    for start in range(0, len(ids), chunk_tokens):
        stop = min(len(ids), start + chunk_tokens)
        piece = ids_tensor[start:stop]
        embeds = text.get_input_embeddings()(piece)
        here = placeholders[start:stop]
        if bool(here.any()):
            embeds = embeds.clone()
            embeds[here] = features[feature_index[start:stop][here]].to(embeds.dtype)
        hidden = text(
            inputs_embeds=embeds[None],
            position_ids=positions[:, None, start:stop].to(device),
            ple_input_ids=piece[None],
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        if on_chunk is not None:
            on_chunk(start, hidden)
    if stats is not None:
        stats.tokens += len(ids)
    return hidden


class ServedReplay:
    """Replays chains through the model as it was served, which is the one
    way a routing replay or a probe should run a chain: its images go through
    the vision tower, and a chain is never run with image placeholders as
    plain tokens. The tower and the processor load with the first chain that
    carries images, so recordings without any cost nothing more."""

    def __init__(self, model: nn.Module, checkpoint: str | Path):
        from training.flash_model import flash_checkpoint_config

        self.model = model
        self.checkpoint = str(checkpoint)
        self.image_token_id = int(flash_checkpoint_config(checkpoint)["image_token_id"])
        self.images = 0  # images put through the tower so far
        self._vision: tuple[Any, Any, dict[str, Any]] | None = None

    def _load_vision(self) -> tuple[Any, Any, dict[str, Any]]:
        if self._vision is None:
            from training.ullm_media import ensure_ullm_media

            ensure_ullm_media()
            from transformers import AutoProcessor
            from ullm.media import processing_identity

            from training.flash_model import attach_flash_multimodal_decoder

            decoder = attach_flash_multimodal_decoder(self.model, self.checkpoint)
            processor = AutoProcessor.from_pretrained(self.checkpoint)
            # The engine's own identity: the local checkpoint, no revision, no
            # processor overrides (training.native_vllm_server), Flash's M-RoPE.
            processing = processing_identity(
                processor,
                model=self.checkpoint,
                revision=None,
                mm_processor_kwargs={},
                model_architecture="Qwen4ExpForConditionalGeneration",
            )
            self._vision = (decoder, processor, processing)
        return self._vision

    def __call__(
        self,
        ids: list[int],
        image_paths: list[str],
        *,
        chunk_tokens: int,
        stats: RouteStats | None = None,
        on_chunk=None,
        what: str = "chain",
    ):
        if not image_paths:
            # Placeholders without images: a trace that lost its media.
            refuse_images_as_text(ids, self.image_token_id, what)
            return replay_chain(self.model, ids, chunk_tokens=chunk_tokens, stats=stats, on_chunk=on_chunk)
        from PIL import Image

        decoder, processor, processing = self._load_vision()
        images = [Image.open(path) for path in image_paths]
        for image in images:
            image.load()
        self.images += len(images)
        return replay_multimodal_chain(
            self.model,
            decoder,
            ids,
            images,
            processor=processor,
            processing=processing,
            chunk_tokens=chunk_tokens,
            stats=stats,
            on_chunk=on_chunk,
        )


def reply_targets(start: int, width: int, total: int, replies: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(hidden row, target position) pairs for the reply tokens a chunk
    ``[start, start + width)`` predicts: row ``i`` predicts position
    ``start + i + 1``."""
    pairs = []
    for begin, end in replies:
        low, high = max(begin, start + 1), min(end, start + width + 1, total)
        pairs.extend((position - 1 - start, position) for position in range(low, high))
    return pairs


class RouterMask:
    """Masks pruned experts out of every layer's router (logits to -inf before
    the softmax): exactly the routing of the physically pruned checkpoint,
    whose router rows are the kept ones. ``set(None)`` restores the full model."""

    def __init__(self, model: nn.Module):
        self.layers = [layer.mlp.gate for layer in model.model.layers]
        self.masks: list[torch.Tensor | None] = [None] * len(self.layers)
        for index, router in enumerate(self.layers):
            original = router.forward

            def masked_forward(hidden_states, _router=router, _index=index, _original=original):
                mask = self.masks[_index]
                if mask is None:
                    return _original(hidden_states)
                hidden_states = hidden_states.reshape(-1, _router.hidden_dim)
                logits = torch.nn.functional.linear(hidden_states, _router.weight).masked_fill(mask, float("-inf"))
                probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
                top_value, indices = torch.topk(probs, _router.top_k, dim=-1)
                if _router.norm_topk_prob:
                    top_value = top_value / top_value.sum(dim=-1, keepdim=True)
                return logits, top_value.to(logits.dtype), indices

            router.forward = masked_forward

    def set(self, plan: dict[str, Any] | None) -> None:
        if plan is None:
            self.masks = [None] * len(self.layers)
            return
        for index, router in enumerate(self.layers):
            kept = torch.tensor(plan["kept"][str(index)], dtype=torch.long, device=router.weight.device)
            mask = torch.ones(router.num_experts, dtype=torch.bool, device=router.weight.device)
            mask[kept] = False
            self.masks[index] = mask


# --- scoring and plans (CPU; numpy) -------------------------------------------


def within_game_weights(scores: dict[str, float], games: dict[str, str], *, beta: float) -> dict[str, float]:
    """``exp(beta * z)`` per rollout, z its ARC score standardised among the
    attempts at the same game (a game where every attempt scored the same
    contributes weight 1 to each: it says nothing about which experts win)."""
    by_game: dict[str, list[str]] = {}
    for run, game in games.items():
        by_game.setdefault(game, []).append(run)
    weights: dict[str, float] = {}
    for runs in by_game.values():
        values = [scores[run] for run in runs]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        for run in runs:
            z = 0.0 if std == 0.0 else (scores[run] - mean) / std
            weights[run] = math.exp(beta * z)
    return weights


def reap_scores(per_run: dict[str, dict[str, Any]], weights: dict[str, float] | None = None):
    """Weighted REAP saliency ``sum_r w_r sum_gnorm_r / sum_r w_r count_r`` per
    layer and expert (plain REAP with every weight 1). Never-routed experts
    score 0."""
    import numpy as np

    runs = sorted(per_run)
    if not runs:
        raise ValueError("no rollouts to score")
    weights = weights or {run: 1.0 for run in runs}
    numerator = sum(weights[run] * per_run[run]["sum_gnorm"] for run in runs)
    denominator = sum(weights[run] * per_run[run]["count"] for run in runs)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def mass_scores(per_run: dict[str, dict[str, Any]], weights: dict[str, float] | None = None):
    """Weighted mass ``sum_r w_r (sum_gnorm_r / tokens_r) / sum_r w_r``: an
    expert's total gate-times-norm contribution per replayed token, frequency
    included (platypus ``expert_prune_plan.py``'s mass+norm; e307's rule adds
    ``cond``, which is REAP's mean). Unlike REAP's mean,
    this is what an outcome weighting moves: an expert used mostly by
    low-scoring rollouts loses mass when they are down-weighted."""
    runs = sorted(per_run)
    if not runs:
        raise ValueError("no rollouts to score")
    weights = weights or {run: 1.0 for run in runs}
    if any(per_run[run]["tokens"] <= 0 for run in runs):
        raise ValueError("every rollout needs replayed tokens")
    total = sum(weights[run] for run in runs)
    return sum(weights[run] * per_run[run]["sum_gnorm"] / per_run[run]["tokens"] for run in runs) / total


def keep_plan(scores, *, keep: int, source: dict[str, Any]) -> dict[str, Any]:
    """The prune tool's plan: per layer, the ``keep`` highest-scoring experts,
    ascending (``scripts/prune_flashnext_experts.py``)."""
    import numpy as np

    layers, experts = scores.shape
    if not 0 < keep < experts:
        raise ValueError("keep must be between 1 and the expert count")
    kept = {}
    for layer in range(layers):
        order = np.argsort(-scores[layer], kind="stable")[:keep]
        kept[str(layer)] = sorted(int(expert) for expert in order)
    return {"num_experts": experts, "kept": kept, **source}
