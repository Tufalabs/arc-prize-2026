"""Optional GDN gated norm and shared/routed expert SwiGLU fusions."""

import torch
from torch.autograd.function import once_differentiable
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from . import gated_pointwise_triton as kernels
from .dtype_contract import floating_inputs


class _NormGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, eps, serving, sigmoid):
        if weight.requires_grad:
            raise ValueError("Fused GDN norm requires frozen norm weights")
        x, z = x.contiguous(), z.contiguous()
        y, r = kernels.norm_gate(x, z, weight, eps, serving, sigmoid)
        ctx.serving, ctx.sigmoid = serving, sigmoid
        ctx.save_for_backward(x, z, weight, r)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        dx, dz = kernels.norm_gate_backward(*ctx.saved_tensors, gradient, ctx.serving, ctx.sigmoid)
        return dx, dz, None, None, None, None


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        if up is not None:
            up = up.contiguous()
        ctx.save_for_backward(gate, up)
        return kernels.swiglu(gate, up, up is None)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        gate, up = ctx.saved_tensors
        return kernels.swiglu_backward(gate, up, gradient, up is None)


@torch.compiler.disable
def norm_gate(x, gate, weight, eps, serving=False, sigmoid=False):
    return _NormGate.apply(x, gate, weight, eps, serving, sigmoid)


@torch.compiler.disable
def swiglu(gate, up=None):
    floating_inputs("swiglu", gate, *(() if up is None else (up,)), cuda=False, same_dtype=True)
    if not gate.is_cuda:
        a, b = gate.chunk(2, -1) if up is None else (gate, up)
        return torch.nn.functional.silu(a) * b
    return _SwiGLU.apply(gate, up)


class TritonGatedNorm(qwen.Qwen4ExpTextRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, activation="silu", *, serving=False):
        super().__init__(hidden_size, eps=eps, activation=activation)
        self.serving = serving

    def forward(self, hidden_states, gate=None):
        # HF's signature defaults gate=None, but its forward always applies the gate.
        if gate is None:
            raise ValueError("TritonGatedNorm: gate is required; Qwen4-Exp's gated norm has no ungated form")
        floating_inputs("TritonGatedNorm", hidden_states, self.weight, gate, cuda=False)
        if not hidden_states.is_cuda:
            if self.serving:
                raise ValueError("TritonGatedNorm serving: requires CUDA tensors")
            return super().forward(hidden_states, gate)
        if self.activation not in ("silu", "swish", "sigmoid"):
            raise ValueError("Fused GDN norm requires SiLU or sigmoid gating")
        return norm_gate(hidden_states, gate, self.weight, self.variance_epsilon,
                         self.serving, self.activation == "sigmoid")


class TritonSharedMLP(qwen.Qwen4ExpTextMLP):
    def forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class _PackQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, heads, value_heads, dim):
        ctx.heads = heads
        return kernels.pack_qkv(x, heads, value_heads, dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk, gv):
        return kernels.pack_qkv_backward(gq, gk, gv, ctx.heads), None, None, None


@torch.compiler.disable
def pack_qkv(x, heads, value_heads, dim):
    return _PackQKV.apply(x, heads, value_heads, dim)


class _ProjectionNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if weight.requires_grad:
            raise ValueError("Fused projection norm requires frozen norm weights")
        y, r = kernels.projection_norm(x, weight, eps)
        ctx.save_for_backward(x, weight, r)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return kernels.projection_norm_backward(*ctx.saved_tensors, gradient), None, None


@torch.compiler.disable
def projection_norm(x, weight, eps):
    return _ProjectionNorm.apply(x, weight, eps)


class TritonProjectionNorm(qwen.Qwen4ExpTextRMSNorm):
    def forward(self, x):
        floating_inputs("TritonProjectionNorm", x, self.weight, cuda=False)
        if not x.is_cuda:
            return super().forward(x)
        if self.group_size is not None or x.ndim != 4 or x.stride(-1) != 1:
            raise ValueError("Fused projection norm requires [batch, tokens, heads, dim] inputs")
        return projection_norm(x, self.weight, self.eps)


class _ScaledAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, adapter, scale):
        ctx.scale = scale
        return kernels.scaled_add(base, adapter, scale)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient * ctx.scale, None


@torch.compiler.disable
def scaled_add(base, adapter, scale):
    return _ScaledAdd.apply(base, adapter, scale)
