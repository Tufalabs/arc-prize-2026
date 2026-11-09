"""Checkpoint every layer; optionally offload a regular subset of boundaries (whole tensors or fixed slabs)."""

from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
    offload_wrapper,
)

from .decision_replay import checkpoint_contexts
from .qsa_selection import selection_contexts
from .qsa_retention import retention_contexts
from .selective_checkpoint import selective_contexts


def resolve_qsa_retention(backend, reuse_selection, requested=None):
    compatible = backend == "flex_qsa" and reuse_selection
    if requested is None:
        return bool(compatible)
    if type(requested) is not bool:
        raise ValueError("retain_qsa_attention must be boolean")
    if requested and not compatible:
        raise ValueError("retain_qsa_attention requires flex_qsa and selection reuse")
    return requested


def should_offload(layer_index, every):
    if type(every) is not int or every < 0:
        raise ValueError("offload_every must be a nonnegative integer")
    return every > 0 and (layer_index + 1) % every == 0


def validate_offload_chunk(mib):
    """Opt-in slab size of offloaded tensors; ``None`` keeps whole-tensor offload."""
    if mib is None:
        return None
    if type(mib) is not int or mib < 1 or mib & (mib - 1):
        raise ValueError("offload_chunk_mib must be a power-of-two number of MiB")
    return mib


def slab_lengths(shape, itemsize, chunk_bytes):
    """Split dimension and slab lengths; every slab but the last has one size."""
    if len(shape) == 0:
        return None, [1]
    dim = max(range(len(shape)), key=shape.__getitem__)
    total = itemsize
    for n in shape:
        total *= n
    if shape[dim] == 0 or total <= chunk_bytes:
        return None, [1]
    rows = max(1, chunk_bytes // (total // shape[dim]))
    lengths = [rows] * (shape[dim] // rows)
    if shape[dim] % rows:
        lengths.append(shape[dim] % rows)
    return dim, lengths


def chunked_save_on_cpu(chunk_bytes, device_type="cuda"):
    """``save_on_cpu(pin_memory=True)`` in fixed-size slabs.

    The pinned host allocator rounds every allocation up to a power of two, so
    one whole saved tensor can hold almost twice its size (a 112k-token
    boundary of 2.1 GiB takes 4 GiB). Slabs of a power-of-two size are stored,
    and reused from the allocator's cache, without that waste; only the last
    slab of a tensor is rounded. The restored tensor is a fresh contiguous
    tensor with the same values, exactly as with the whole-tensor copy.
    """
    import torch
    from torch.autograd.graph import saved_tensors_hooks

    pin = getattr(torch, device_type, torch.cuda).is_available()

    def pack(tensor):
        if tensor.is_sparse:
            dim, lengths = None, [1]
        else:
            dim, lengths = slab_lengths(tuple(tensor.shape), tensor.element_size(), chunk_bytes)
        if dim is None:
            packed = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
                                 pin_memory=pin and not tensor.is_sparse)
            packed.copy_(tensor)
            return tensor.device, None, None, [packed]
        slabs, start = [], 0
        for length in lengths:
            piece = tensor.narrow(dim, start, length)
            slab = torch.empty(piece.shape, dtype=tensor.dtype, pin_memory=pin)
            slab.copy_(piece)
            slabs.append(slab)
            start += length
        return tensor.device, dim, tuple(tensor.shape), slabs

    def unpack(packed):
        device, dim, shape, slabs = packed
        if dim is None:
            return slabs[0].to(device, non_blocking=pin)
        out = torch.empty(shape, dtype=slabs[0].dtype, device=device)
        start = 0
        for slab in slabs:
            out.narrow(dim, start, slab.shape[dim]).copy_(slab, non_blocking=pin)
            start += slab.shape[dim]
        return out

    return saved_tensors_hooks(pack, unpack)


def chunked_offload_wrapper(module, chunk_bytes):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper

    class ChunkedOffloadWrapper(ActivationWrapper):
        def forward(self, *args, **kwargs):
            with chunked_save_on_cpu(chunk_bytes):
                return self._checkpoint_wrapped_module(*args, **kwargs)

    return ChunkedOffloadWrapper(module)


def checkpoint_layer(
    layer, layer_index, every, reuse_qsa_selection=False, retain_qsa_attention=False,
    checkpoint_policy="full", offload_chunk_mib=None,
):
    if checkpoint_policy == "small_gemm" and reuse_qsa_selection:
        raise ValueError("checkpoint_layer: checkpoint_policy='small_gemm' with reuse_qsa_selection=True is unsupported")
    enabled = should_offload(layer_index, every)
    chunk = validate_offload_chunk(offload_chunk_mib)
    context_fn = selection_contexts if reuse_qsa_selection else checkpoint_contexts
    if retain_qsa_attention:
        if not reuse_qsa_selection:
            raise ValueError("QSA attention retention requires selection reuse")
        context_fn = retention_contexts
    if checkpoint_policy not in ("full", "small_gemm"):
        raise ValueError(f"checkpoint_layer: checkpoint_policy={checkpoint_policy!r}; allowed values are full, small_gemm")
    if checkpoint_policy == "small_gemm":
        context_fn = partial(selective_contexts, context_fn)
    wrapped = checkpoint_wrapper(
        layer,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=True,
        context_fn=context_fn,
    )
    if not enabled:
        return wrapped
    return offload_wrapper(wrapped) if chunk is None else chunked_offload_wrapper(wrapped, chunk << 20)
