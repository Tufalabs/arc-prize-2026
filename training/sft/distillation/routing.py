"""Observe original forwards only; recomputation must never increment counters."""

import torch
import torch.distributed as dist


class RouteRecorder:
    def __init__(self, layers, experts, device):
        self.counts = torch.zeros(2, layers, experts, dtype=torch.int64, device=device)
        self.mass = torch.zeros(2, layers, experts, dtype=torch.float32, device=device)
        self.enabled = False
        self.prompt_len = 0
        self.completion_mask = None
        self.context_mask = None

    def begin(self, prompt_len, context_mask=None):
        self.counts.zero_()
        self.mass.zero_()
        if torch.is_tensor(prompt_len):
            if prompt_len.dtype != torch.bool or prompt_len.ndim != 1:
                raise ValueError(
                    "Completion prediction mask must be one-dimensional bool"
                )
            self.completion_mask = prompt_len
            self.context_mask = context_mask
        else:
            self.prompt_len = prompt_len
            self.completion_mask = None
            self.context_mask = None
        self.enabled = True

    @torch.no_grad()
    def record(self, layer, indices, weights):
        if not self.enabled:
            return
        if self.completion_mask is None:
            views = [
                (indices, weights),
                (indices[self.prompt_len - 1 : -1], weights[self.prompt_len - 1 : -1]),
            ]
        else:
            if len(indices) != len(self.completion_mask):
                raise ValueError("Packed routing mask length mismatch")
            context = (
                torch.ones_like(self.completion_mask)
                if self.context_mask is None
                else self.context_mask
            )
            views = [
                (indices[context], weights[context]),
                (indices[self.completion_mask], weights[self.completion_mask]),
            ]
        for scope, (idx, val) in enumerate(views):
            idx = idx.reshape(-1)
            self.counts[scope, layer].scatter_add_(0, idx, torch.ones_like(idx))
            self.mass[scope, layer].scatter_add_(
                0, idx, val.detach().reshape(-1).float()
            )

    def finish(self):
        self.enabled = False

    def report(self, distributed=True):
        if self.enabled:
            raise RuntimeError("finish forward capture before reduction")
        counts, mass = self.counts.clone(), self.mass.clone()
        if distributed:
            dist.all_reduce(counts)
            dist.all_reduce(mass)
        return {
            scope: {
                "selection_counts": counts[i].cpu().tolist(),
                "routing_mass": mass[i].cpu().tolist(),
            }
            for i, scope in enumerate(["all_context", "completion_predictions"])
        }


def attach(model, recorder):
    gates = [
        module for name, module in model.named_modules() if name.endswith("mlp.gate")
    ]
    if len(gates) != recorder.counts.shape[1]:
        raise ValueError("router module census mismatch")
    for i, gate in enumerate(gates):
        gate.route_recorder, gate.route_layer = recorder, i
