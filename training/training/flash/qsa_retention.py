"""Keep only attention-core saved tensors across whole-layer checkpoint replay.

Inner saved-tensor hooks bypass both checkpoint discard and CPU offloading.
The original attention autograd node performs backward; replay returns a detached
leaf with matching requires_grad so surrounding operations save identical tensors.
No graph or cache is shared between checkpoint invocations.
"""

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from .qsa_selection import selection_contexts

_active = ContextVar("qsa_attention_retention", default=None)


@dataclass
class _State:
    entries: list = field(default_factory=list)


@dataclass
class _Invocation:
    state: _State
    replay: bool
    cursor: int = 0


class _Scope:
    def __init__(self, state, replay, selection):
        self.state, self.replay, self.selection = state, replay, selection
        self.stack = []

    def __enter__(self):
        stack = ExitStack()
        stack.enter_context(self.selection)
        token = _active.set(_Invocation(self.state, self.replay))
        stack.callback(_active.reset, token)
        self.stack.append(stack)
        return self

    def __exit__(self, *args):
        return self.stack.pop().__exit__(*args)


def retention_contexts():
    state = _State()
    forward, replay = selection_contexts()
    return _Scope(state, False, forward), _Scope(state, True, replay)


def _unpack(saved):
    tensor, version = saved
    if tensor._version != version:
        raise RuntimeError("Retained QSA backward tensor modified in place")
    return tensor


def retain_attention(owner, inputs, compute):
    """Return output, replay flag, and unique saved-storage bytes (diagnostic)."""
    active = _active.get()
    if active is None or not torch.is_grad_enabled():
        return compute(), False, 0
    signature = (id(owner), tuple((tuple(t.shape), t.dtype, t.device) for t in inputs))
    cursor = active.cursor
    active.cursor += 1
    if active.replay:
        if cursor >= len(active.state.entries):
            raise RuntimeError("QSA attention replay has no retained forward")
        expected, output, version, needs_grad, nbytes = active.state.entries[cursor]
        if expected != signature:
            raise RuntimeError("QSA attention replay invocation mismatch")
        if output._version != version:
            raise RuntimeError("Retained QSA output modified in place")
        return output.detach().requires_grad_(needs_grad), True, nbytes
    if cursor != len(active.state.entries):
        raise RuntimeError("QSA attention forward context reused")
    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return tensor.detach(), tensor._version

    with torch.autograd.graph.saved_tensors_hooks(pack, _unpack):
        output = compute()
    saved = output.detach()
    storage = saved.untyped_storage()
    storages[(saved.device, storage.data_ptr())] = storage.nbytes()
    nbytes = sum(storages.values())
    active.state.entries.append(
        (signature, saved, saved._version, output.requires_grad, nbytes)
    )
    return output, False, nbytes
