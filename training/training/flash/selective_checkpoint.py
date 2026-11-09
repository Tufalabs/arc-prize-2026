"""Selective replay of pointwise work with bounded-width GEMM retention."""

from contextlib import ExitStack

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


def small_gemm_policy(ctx, op, *args, **kwargs):
    # Keep low-rank adapter/GDN gate projections, not multi-GB expert or
    # hyper-connection activations. All other operations remain recomputable.
    if op == torch.ops.aten.mm.default and args[1].shape[-1] <= 128:
        return CheckpointPolicy.MUST_SAVE
    if op == torch.ops.aten.addmm.default and args[2].shape[-1] <= 128:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


class _JoinedScope:
    def __init__(self, scopes):
        self.scopes = scopes
        self.stacks = []

    def __enter__(self):
        stack = ExitStack()
        try:
            for scope in self.scopes:
                stack.enter_context(scope)
        except BaseException:
            stack.close()
            raise
        self.stacks.append(stack)
        return self

    def __exit__(self, *args):
        return self.stacks.pop().__exit__(*args)


class _SingleReplay:
    def __init__(self, scope):
        self.scope = scope
        self.used = False

    def __enter__(self):
        if self.used:
            raise RuntimeError("checkpoint_policy='small_gemm' does not support repeated backward with retain_graph=True; use 'full'")
        self.used = True
        return self.scope.__enter__()

    def __exit__(self, *args):
        return self.scope.__exit__(*args)


def selective_contexts(existing=None):
    forward, replay = create_selective_checkpoint_contexts(small_gemm_policy)
    replay = _SingleReplay(replay)
    if existing is None:
        return forward, replay
    original_forward, original_replay = existing()
    return _JoinedScope((original_forward, forward)), _JoinedScope((original_replay, replay))
