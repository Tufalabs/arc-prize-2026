"""Fuse only expert weighting/permutation; retain eager gradient reductions."""

import torch


@torch.compile(fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
def _weighted_permutation(x, weight, inverse):
    return x[inverse] * weight[inverse, None]


class WeightedPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, permutation, inverse):
        ctx.save_for_backward(x, weight, permutation)
        if x.is_cuda:
            return _weighted_permutation(x, weight, inverse)
        return x[inverse] * weight[inverse, None]

    @staticmethod
    def backward(ctx, gradient):
        x, weight, permutation = ctx.saved_tensors
        # Unique permutation indices preserve the eager per-row arithmetic.
        # Bound the gathered-gradient temporary. Keep the final reduction at its
        # original full shape: CUDA chooses different rounding for short tails.
        dx = (
            torch.empty(
                x.shape,
                device=x.device,
                dtype=torch.promote_types(gradient.dtype, weight.dtype),
            )
            if ctx.needs_input_grad[0]
            else None
        )
        product = (
            torch.empty(
                x.shape,
                device=x.device,
                dtype=torch.promote_types(gradient.dtype, x.dtype),
            )
            if ctx.needs_input_grad[1]
            else None
        )
        for start in range(0, x.shape[0], 8192):
            end = min(start + 8192, x.shape[0])
            sorted_gradient = gradient[permutation[start:end]]
            if dx is not None:
                dx[start:end] = sorted_gradient * weight[start:end, None]
            if product is not None:
                product[start:end] = sorted_gradient * x[start:end]
        dw = product.sum(dim=-1) if product is not None else None
        return dx, dw, None, None


def weighted_permutation(x, weight, permutation, inverse):
    return WeightedPermutation.apply(x, weight, permutation, inverse)
