"""Fuse RMSNorm's final multiplications/cast; leave statistics and reductions eager."""

import torch


@torch.compile(fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
def _tail(x, rstd, gain, dtype):
    return ((x * rstd) * gain).to(dtype)


class NormTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rstd, gain, dtype):
        if gain.requires_grad:
            raise ValueError("NormTail: requires frozen gain weights")
        ctx.save_for_backward(x, rstd, gain)
        return (
            _tail(x, rstd, gain, dtype) if x.is_cuda else ((x * rstd) * gain).to(dtype)
        )

    @staticmethod
    def backward(ctx, gradient):
        x, rstd, gain = ctx.saved_tensors
        dnorm = gradient.float() * gain
        dx = dnorm * rstd if ctx.needs_input_grad[0] else None
        drstd = (dnorm * x).sum(-1, keepdim=True) if ctx.needs_input_grad[1] else None
        return dx, drstd, None, None


def grouped_norm_forward(self, x):
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], -1, self.group_size)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    gain = (1.0 + self.weight.float()).reshape(-1, self.group_size)
    return NormTail.apply(xf, rstd, gain, x.dtype).reshape(shape)
