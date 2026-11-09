"""Public padded-batch layout for the shared Flash model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DocumentSpan:
    start: int
    tokens: int
    padded_tokens: int

    @property
    def stop(self):
        return self.start + self.tokens

    @property
    def padded_stop(self):
        return self.start + self.padded_tokens


@dataclass(frozen=True)
class PackedBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    spans: tuple[DocumentSpan, ...]
    unpack_indices: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def packed_lengths(self):
        return tuple(span.padded_tokens for span in self.spans)

    def pack(self, values, padding_value=0):
        """Pack token-aligned values without cutting any document or image run."""
        if values.shape[:2] != self.unpack_indices.shape:
            raise ValueError("Values must match the public batch layout")
        parts = []
        for row, span in enumerate(self.spans):
            tail = values.new_ones((span.padded_tokens - span.tokens, *values.shape[2:]))
            parts.extend((values[row, :span.tokens], tail * padding_value))
        return torch.cat(parts, dim=0)[None]

    def unpack(self, hidden):
        if hidden.ndim < 2 or hidden.shape[0] != 1:
            raise ValueError("Packed output must have one physical token stream")
        if hidden.shape[1] != self.input_ids.shape[1]:
            raise ValueError("Packed output length changed")
        indices = self.unpack_indices.clamp_min(0).flatten()
        result = (
            hidden[0]
            .index_select(0, indices)
            .reshape(*self.unpack_indices.shape, *hidden.shape[2:])
        )
        mask = self.attention_mask.reshape(
            *self.attention_mask.shape,
            *((1,) * (hidden.ndim - 2)),
        )
        return result * mask.to(result.dtype)


def positions_reset(position_ids, documents):
    """True when positions step backwards inside any document.

    ``position_ids`` is ``[B, T]`` (text) or ``[3, B, T]`` (M-RoPE: its first
    axis is the temporal index, which never decreases inside one document even
    though an image's row and column axes do). ``documents`` yields
    ``(row, start, stop)`` token ranges, one per document."""
    temporal = position_ids[0] if position_ids.ndim == 3 else position_ids
    for row, start, stop in documents:
        segment = temporal[row, start:stop]
        if bool((segment[1:] < segment[:-1]).any()):
            return True
    return False


def pack_padded_batch(
    input_ids,
    attention_mask=None,
    position_ids=None,
    *,
    pad_token_id=0,
    eos_token_id=0,
    alignment=128,
):
    """Convert a conventional right-padded batch to one isolated token stream."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a two-dimensional tensor")
    batch, sequence = input_ids.shape
    if batch < 1 or sequence < 1 or type(alignment) is not int or alignment < 1:
        raise ValueError("Batch, sequence and alignment must be positive")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a nonnegative integer")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif (
        not isinstance(attention_mask, torch.Tensor)
        or tuple(attention_mask.shape) != tuple(input_ids.shape)
        or attention_mask.device != input_ids.device
    ):
        raise ValueError("attention_mask must match input_ids")
    else:
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one")
        attention_mask = attention_mask.bool()
    if torch.any(attention_mask[:, 1:] & ~attention_mask[:, :-1]):
        raise ValueError("Only right-padded attention masks are supported")
    lengths = attention_mask.sum(1).tolist()
    if any(length < 1 for length in lengths):
        raise ValueError("Every batch row must contain at least one token")
    if position_ids is not None and (
        not isinstance(position_ids, torch.Tensor)
        or tuple(position_ids.shape) not in (tuple(input_ids.shape), (3, batch, sequence))
        or position_ids.device != input_ids.device
    ):
        raise ValueError("position_ids must have shape [B, T] or [3, B, T] and match input_ids")
    if position_ids is not None and positions_reset(
        position_ids, ((row, 0, length) for row, length in enumerate(lengths))
    ):
        raise ValueError("position_ids reset inside a padded row; a document boundary needs its own row")

    parts, positions, spans, boundaries = [], [], [], [0]
    unpack_indices = torch.full(
        input_ids.shape, -1, dtype=torch.long, device=input_ids.device
    )
    start = 0
    for row, length in enumerate(lengths):
        # Reserve at least one separator and fill the complete tail with EOS.
        # Qwen4-Exp PLE resets n-gram history at EOS; ordinary pad tokens would
        # otherwise carry the previous row's context into the next document.
        padded = ((length + 1 + alignment - 1) // alignment) * alignment
        part = torch.full(
            (padded,), eos_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        part[:length].copy_(input_ids[row, :length])
        parts.append(part)
        if position_ids is None:
            local_positions = torch.arange(
                padded, dtype=torch.long, device=input_ids.device
            )
        else:
            active_positions = position_ids[..., row, :length].to(torch.long)
            tail = torch.arange(
                1,
                padded - length + 1,
                dtype=torch.long,
                device=input_ids.device,
            )
            if position_ids.ndim == 3:
                tail = (active_positions.max() + tail).expand(3, -1)
            else:
                tail = active_positions[-1] + tail
            local_positions = torch.cat([active_positions, tail], dim=-1)
        positions.append(local_positions)
        spans.append(DocumentSpan(start, length, padded))
        unpack_indices[row, :length] = start + torch.arange(
            length, device=input_ids.device
        )
        start += padded
        boundaries.append(start)
    return PackedBatch(
        input_ids=torch.cat(parts)[None],
        position_ids=torch.cat(positions, dim=-1).unsqueeze(-2),
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=input_ids.device),
        spans=tuple(spans),
        unpack_indices=unpack_indices,
        attention_mask=attention_mask,
    )
