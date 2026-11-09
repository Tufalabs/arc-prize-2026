"""Opt-in replay of discrete Flash decisions for one forward/backward graph.

Use ``with tape.record():`` around a reference forward AND backward, then
``with tape.replay():`` around each candidate forward AND backward. Create a
fresh DecisionTape per input/microbatch. Layer IDs are stable across model
instances; repeated visits to a layer (activation checkpoint recomputation)
reuse its recorded indices. Nested contexts restore their predecessor.

QSA stores int32 block IDs including -1 padding. Expert indices use int16 when
possible (48 layers * 10 experts * 2 bytes = 960 bytes/token), otherwise int32.
Only decisions are retained, never logits, activations or autograd graphs.
Replay gathers weights from CURRENT router probabilities, preserving gradients
and the caller's norm_topk_prob normalization. Packed lengths are part of each
entry's contract: equal token counts with different boundaries cannot replay.

The tape is explicit, process-local state, not a model/checkpoint manifest option.
The Flash checkpoint contexts bind the tape to their graph, including backward
on autograd worker threads or after the outer recording scope ends. Third-party
checkpoint wrappers must compose checkpoint_contexts(), even if the outer tape
context remains active through backward. Entries
can be inspected for diagnostics; in-place edits are rejected on subsequent use.
"""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

_active = ContextVar("flash_decision_replay", default=None)


@dataclass
class Decision:
    signature: tuple
    indices: torch.Tensor
    version: int


@dataclass
class DecisionTape:
    entries: dict[tuple[str, int], Decision] = field(default_factory=dict)

    @contextmanager
    def record(self):
        """Record a fresh graph, reusing decisions during its recomputation."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("DecisionTape.record: inference_mode=True is unsupported")
        if self.entries:
            raise RuntimeError("Record requires a fresh decision tape")
        token = _active.set((self, True))
        try:
            yield self
        finally:
            _active.reset(token)

    @contextmanager
    def replay(self):
        """Force recorded indices; arithmetic and router weights remain live."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("DecisionTape.replay: inference_mode=True is unsupported")
        token = _active.set((self, False))
        try:
            yield self
        finally:
            _active.reset(token)

    @property
    def nbytes(self):
        return sum(
            e.indices.numel() * e.indices.element_size() for e in self.entries.values()
        )


class _CheckpointReplayScope:
    """Re-entrant context owned by one checkpoint graph, not the calling thread."""

    def __init__(self, active):
        self.active = None if active is None else (active[0], False)
        self.tokens = []

    def __enter__(self):
        self.tokens.append(_active.set(self.active))
        return self

    def __exit__(self, *_exc):
        _active.reset(self.tokens.pop())
        return False


def checkpoint_contexts():
    """Capture replay state for torch.utils.checkpoint's context_fn.

    CUDA autograd may recompute on a worker thread; ContextVars alone do not
    propagate there. The forward scope is unchanged. A captured tape always
    replays during recomputation, even when the enclosing scope was recording.
    """
    return nullcontext(), _CheckpointReplayScope(_active.get())


def _layout(lengths, tokens):
    if lengths is None:
        return None
    lengths = tuple(int(n) for n in lengths)
    if any(n <= 0 for n in lengths) or sum(lengths) != tokens:
        raise ValueError("Invalid packed decision-replay layout")
    return lengths


def _indices(kind, layer, signature, compute, dtype, device):
    tape, recording = _active.get()
    if layer is None:
        raise ValueError("Decision replay requires a stable layer index")
    key = (kind, int(layer))
    entry = tape.entries.get(key)
    if entry is None:
        if not recording:
            raise RuntimeError(f"Missing replay decision: {key}")
        with torch.no_grad():
            indices = compute().detach().to(dtype=dtype).clone()
        entry = Decision(signature, indices, indices._version)
        tape.entries[key] = entry
    if entry.signature != signature:
        raise RuntimeError(f"Decision replay layout/shape mismatch: {key}")
    if entry.indices._version != entry.version:
        raise RuntimeError(f"Replay decision modified in place: {key}")
    return entry.indices.to(device=device)


def qsa_blocks(indexer, hidden, compute, packed_lengths=None):
    """Apply optional QSA replay inside the checkpoint-selection cache."""
    if _active.get() is None:
        return compute()
    signature = (
        tuple(hidden.shape),
        _layout(packed_lengths, hidden.shape[1]),
        indexer.compress_ratio,
        indexer.token_budget,
        indexer.block_topk,
    )
    return _indices(
        "qsa", indexer.layer_idx, signature, compute, torch.int32, hidden.device
    )


def expert_topk(probs, topk, layer=None, packed_lengths=None):
    """Return top-k weights/indices, gathering live weights under replay."""
    if _active.get() is None:
        return probs.topk(topk, -1)
    signature = (tuple(probs.shape), _layout(packed_lengths, probs.shape[0]), topk)
    dtype = torch.int16 if probs.shape[-1] <= 32768 else torch.int32
    indices = _indices(
        "experts",
        layer,
        signature,
        lambda: probs.topk(topk, -1).indices,
        dtype,
        probs.device,
    ).long()
    # Both recording and replay use gather, including checkpoint recomputation.
    # Using differentiable topk during recording would save a different graph.
    return probs.gather(-1, indices), indices


def reject_replay(component, remedy):
    """Raise inside an active tape for a component that never consults it."""
    if _active.get() is not None:
        raise RuntimeError(f"{component}: DecisionTape is unsupported; {remedy}")
