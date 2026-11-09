"""Retain compact QSA selections within one non-reentrant checkpoint graph.

The context factory runs once per checkpoint invocation. No cache is stored on
the model, so outstanding forwards/microbatches cannot overwrite each other.
Only integer indices survive; input activations and selector graphs do not.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from .decision_replay import checkpoint_contexts

_active = ContextVar("qsa_checkpoint_selection", default=None)


@dataclass
class _State:
    entries: list = field(default_factory=list)


@dataclass
class _Invocation:
    state: _State
    recompute: bool
    cursor: int = 0


class _Scope:
    def __init__(self, state, recompute, decisions):
        self.state, self.recompute = state, recompute
        self.decisions = decisions
        self.tokens = []

    def __enter__(self):
        self.decisions.__enter__()
        self.tokens.append(_active.set(_Invocation(self.state, self.recompute)))
        return self

    def __exit__(self, *_exc_info):
        _active.reset(self.tokens.pop())
        self.decisions.__exit__(*_exc_info)
        # Non-reentrant checkpoint may stop replay early. Retain entries for
        # retain_graph/repeated backward; the checkpoint graph owns their lifetime.
        return False


def selection_contexts():
    state = _State()
    forward, replay = checkpoint_contexts()
    return _Scope(state, False, forward), _Scope(state, True, replay)


def select_blocks(indexer, hidden, compute):
    """Return (indices, reused), preserving the original selector arithmetic."""
    active = _active.get()
    if active is None:
        return compute(), False
    signature = (
        id(indexer),
        tuple(hidden.shape),
        hidden.dtype,
        hidden.device,
        indexer.compress_ratio,
        indexer.token_budget,
    )
    cursor = active.cursor
    active.cursor += 1
    if active.recompute:
        if cursor >= len(active.state.entries):
            raise RuntimeError("QSA checkpoint replay has no forward selection")
        expected, indices, version = active.state.entries[cursor]
        if expected != signature:
            raise RuntimeError("QSA checkpoint selection invocation mismatch")
        if indices._version != version:
            raise RuntimeError("QSA checkpoint selection was modified in place")
        return indices, True
    if cursor != len(active.state.entries):
        raise RuntimeError("QSA checkpoint forward context was reused")
    indices = compute()
    if (
        indices.dtype != torch.int32
        or indices.requires_grad
        or indices.device != hidden.device
        or tuple(indices.shape) != (hidden.shape[1], indexer.block_topk)
    ):
        raise RuntimeError("QSA checkpoint cache requires compact int32 block IDs")
    active.state.entries.append((signature, indices, indices._version))
    return indices, False
