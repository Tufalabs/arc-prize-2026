"""Opt-in expert_lora_sharded: routed-expert LoRA masters split along the rank across ranks.

Without it every rank holds all routed-expert adapters (FP32 masters, their
gradients and optimizer state), so their memory per rank does not fall with the
world size. With it each rank keeps one contiguous slice of every adapter's rank
dimension (``lora_*_A`` ``[experts, rank, in]`` along dim 1, ``lora_*_B``
``[experts, out, rank]`` along dim 2): rank ``w`` holds ranks
``w*r/W .. (w+1)*r/W - 1``. Weights, gradients and optimizer state per rank fall
by the world size W.

Each use of an adapter in a layer's forward, and in that layer's checkpoint
replay, all-gathers the full FP32 master in rank order (the values the
replicated layout holds), and the arithmetic after it is the replicated
layout's. Its backward all-reduces (sums) the full FP32 gradient, as the
replicated layout reduces that parameter's gradient, and keeps this rank's
slice of the sum. Every rank must run the same layers in the same order: the
trainer's padding slots and validation fillers already make every rank run the
same forwards and backwards.

Ported from the distillation trainer (claude/expert-lora-fsdp 6aa9b2a2, as run
in claude/kimi-sft-baseline 0942d180); the checkpoint helpers are Flash SFT's.
"""

import torch
import torch.distributed as dist

# Dimension of the LoRA rank in lora_<name>_A / lora_<name>_B.
SHARD_DIM = {"A": 1, "B": 2}


def validate_expert_lora_sharded(value, expert_lora):
    if type(value) is not bool:
        raise ValueError("expert_lora_sharded must be boolean")
    if value and expert_lora is None:
        raise ValueError("expert_lora_sharded requires routed expert LoRA")


def is_rank_shard(parameter):
    return getattr(parameter, "_lora_rank_shard", None) is not None


def shard_expert_adapters(adapters, rank, world):
    """Keep this rank's slice of each routed adapter, in place (same Parameter objects)."""
    for name, p in adapters.items():
        dim = SHARD_DIM[name[-1]]
        size = p.shape[dim]
        if size % world:
            raise ValueError("expert_lora_sharded requires the LoRA rank to divide by the world size")
        width = size // world
        with torch.no_grad():
            p.data = p.data.narrow(dim, rank * width, width).clone()
        p._lora_rank_shard = (dim, world)
    return adapters


def _gather(shard, dim, world):
    moved = shard.detach().movedim(dim, 0).contiguous()
    if dist.get_backend() == "gloo":
        # CPU tests: the same rank-ordered concatenation.
        parts = [torch.empty_like(moved) for _ in range(world)]
        dist.all_gather(parts, moved)
        full = torch.cat(parts)
    else:
        full = moved.new_empty((world * moved.shape[0], *moved.shape[1:]))
        dist.all_gather_into_tensor(full, moved)
    return full.movedim(0, dim).contiguous()


class _GatherRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, dim, world):
        ctx.dim, ctx.world = dim, world
        return _gather(shard, dim, world)

    @staticmethod
    def backward(ctx, grad):
        # The replicated layout's reduction of this full-shape FP32 gradient (a sum), then this rank's slice.
        full = grad.contiguous().clone()
        dist.all_reduce(full)
        width = full.shape[ctx.dim] // ctx.world
        return full.narrow(ctx.dim, dist.get_rank() * width, width).contiguous(), None, None


def adapter_weight(module, name):
    """The full FP32 master of ``module.<name>``: the parameter itself, or its gathered slices."""
    p = getattr(module, name)
    shard = getattr(p, "_lora_rank_shard", None)
    if shard is None:
        return p
    return _GatherRank.apply(p, *shard)


@torch.no_grad()
def full_value(parameter):
    """The full tensor of a rank-sharded parameter (a collective: every rank calls it), else the parameter."""
    if not is_rank_shard(parameter):
        return parameter.detach()
    return _gather(parameter, *parameter._lora_rank_shard)


def rank_slice(full, parameter, rank):
    """This rank's slice of a full adapter tensor, shaped as the sharded ``parameter``."""
    dim, world = parameter._lora_rank_shard
    width = full.shape[dim] // world
    return full.narrow(dim, rank * width, width)


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True):
    """torch.nn.utils.clip_grad_norm_ for replicated and rank-sharded parameters together.

    The norm is the full gradient's: the replicated gradients' norms and the shards'
    squared norms summed over ranks. The coefficient and the scaling are clip_grad_norm_'s.
    Every rank calls it.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(p.grad) for p in parameters if not is_rank_shard(p)]
    shards = [torch.linalg.vector_norm(p.grad) for p in parameters if is_rank_shard(p)]
    squares = torch.linalg.vector_norm(torch.stack(norms)).square() if norms else 0.0
    if shards:
        part = torch.linalg.vector_norm(torch.stack(shards)).square()
        dist.all_reduce(part)
        squares = squares + part
    total = torch.as_tensor(squares).sqrt()
    if error_if_nonfinite and not torch.isfinite(total):
        raise RuntimeError("The total norm of gradients is non-finite, so it cannot be clipped")
    coefficient = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for p in parameters:
        p.grad.mul_(coefficient.to(p.grad.device))
    return total


def split_optimizer_state(state, indices):
    """(state_dict without the sharded entries, the sharded entries by index)."""
    shared = {"state": {}, "param_groups": state["param_groups"]}
    own = {}
    for index, value in state["state"].items():
        (own if index in indices else shared["state"])[index] = value
    return shared, own


def merge_optimizer_state(shared, own):
    state = dict(shared["state"])
    if set(state) & set(own):
        raise ValueError("sharded optimizer state overlaps the shared state")
    state.update(own)
    return {"state": dict(sorted(state.items())), "param_groups": shared["param_groups"]}
