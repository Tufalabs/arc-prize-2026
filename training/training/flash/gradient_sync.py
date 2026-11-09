"""Optional bucketed reduction for replicated Flash adapter gradients."""

from contextlib import contextmanager

import torch
import torch.distributed as dist


class ReplicatedGradientReducer:
    """Issue buckets in a fixed backward order as their gradients become ready.

    Supply nonempty buckets in backward order (last decoder first). Every
    parameter must receive a gradient on every rank each step. Call finish()
    after backward and before clipping/optimizer updates, then zero gradients.
    For accumulation, put all but the final microbatch backward in no_sync().
    This reducer is for replicated adapters excluded from FSDP gradient sharding.

    A readiness error poisons the instance for the rest of the process: ranks
    are desynchronized after it and in-flight reductions are abandoned, so
    there is deliberately no reset; a new instance would only hang on the next
    collective.
    """

    def __init__(self, buckets):
        self.buckets = [tuple(bucket) for bucket in buckets if bucket]
        parameters = [p for bucket in self.buckets for p in bucket]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("Gradient buckets must not share parameters")
        self.ready = [0] * len(self.buckets)
        self.next_bucket = 0
        self.pending = []
        self.hooks = []
        self.sync_depth = 0
        self.seen = set()
        self.failed = False
        for index, bucket in enumerate(self.buckets):
            for parameter in bucket:
                self.hooks.append(parameter.register_post_accumulate_grad_hook(self._hook(index)))

    def _hook(self, index):
        def ready(parameter):
            if self.failed:
                raise RuntimeError("ReplicatedGradientReducer: a previous accumulation error invalidated this step")
            if index < self.next_bucket or id(parameter) in self.seen:
                self.failed = True
                raise RuntimeError(
                    f"ReplicatedGradientReducer: gradient contribution to bucket {index} after readiness/reduction; "
                    "use no_sync() for all but the final microbatch, then finish() and zero gradients"
                )
            if self.sync_depth:
                return
            self.seen.add(id(parameter))
            self.ready[index] += 1
            while (self.next_bucket < len(self.buckets)
                   and self.ready[self.next_bucket] == len(self.buckets[self.next_bucket])):
                bucket = self.buckets[self.next_bucket]
                flat = torch.cat([p.grad.reshape(-1) for p in bucket])
                work = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=True)
                self.pending.append((bucket, flat, work))
                self.next_bucket += 1
        return ready

    @contextmanager
    def no_sync(self):
        """Accumulate local microbatches; run the final backward outside this scope."""
        if self.next_bucket or self.seen or self.failed:
            raise RuntimeError("ReplicatedGradientReducer.no_sync: reductions/readiness already started")
        self.sync_depth += 1
        try:
            yield
        finally:
            self.sync_depth -= 1

    def finish(self):
        if self.sync_depth or self.failed:
            raise RuntimeError("ReplicatedGradientReducer.finish: active no_sync or invalidated accumulation step")
        if self.next_bucket != len(self.buckets):
            raise RuntimeError("Every replicated gradient bucket must be complete")
        for bucket, flat, work in self.pending:
            work.wait()
            offset = 0
            for parameter in bucket:
                size = parameter.numel()
                parameter.grad.copy_(flat[offset:offset + size].view_as(parameter))
                offset += size
        self.pending.clear()
        self.seen.clear()
        self.ready = [0] * len(self.buckets)
        self.next_bucket = 0

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
