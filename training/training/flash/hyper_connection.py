"""Opt-in hyper-connection fusion, with ordinary PyTorch GEMMs."""

import torch
from torch.autograd.function import once_differentiable
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from . import hyper_connection_triton as kernels
from .dtype_contract import floating_inputs, low_precision_inputs, linear_inputs


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, groups, eps, serving):
        if weight.requires_grad:
            raise ValueError("Fused grouped norm requires frozen norm weights")
        kernels._norm_shapes("HC norm", x, weight, groups)
        x = x.contiguous()
        y, r = kernels.norm(x, weight, groups, eps, serving)
        ctx.save_for_backward(x, weight, r)
        ctx.groups = groups
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return kernels.norm_backward(*ctx.saved_tensors, gradient, ctx.groups), None, None, None, None


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, groups, inject, serving):
        ctx.save_for_backward(x)
        ctx.groups, ctx.inject, ctx.serving = groups, inject, serving
        return kernels.activation(x, groups, inject, serving)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return kernels.activation_backward(ctx.saved_tensors[0], gradient, ctx.groups, ctx.inject, ctx.serving), None, None, None


class _Mix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, groups, serving):
        ctx.save_for_backward(x, up)
        ctx.groups, ctx.serving = groups, serving
        return kernels.mix(x, up, groups, serving)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        dx, du = kernels.mix_backward(*ctx.saved_tensors, gradient, ctx.groups, ctx.serving)
        return dx, du, None, None


class _Inject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, weights, serving):
        ctx.save_for_backward(h, weights)
        ctx.serving = serving
        return kernels.inject(x, h, weights, serving)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return (*kernels.inject_backward(gradient, *ctx.saved_tensors, ctx.serving), None)


class _InjectNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, weights, gain, eps, serving):
        if gain.requires_grad:
            raise ValueError("Fused grouped norm requires frozen norm weights")
        y, z, r = kernels.inject_norm(x, h, weights, gain, eps, serving)
        ctx.save_for_backward(y, h, weights, gain, r)
        ctx.serving = serving
        return y, z

    @staticmethod
    @once_differentiable
    def backward(ctx, gy, gz):
        y, h, weights, gain, r = ctx.saved_tensors
        gradient = kernels.inject_norm_backward(y, gain, r, gy, gz, weights.shape[-1])
        return (*kernels.inject_backward(gradient, h, weights, ctx.serving), None, None, None)


@torch.compiler.disable
def grouped_norm(x, weight, groups, eps, serving=False):
    return _Norm.apply(x, weight, groups, eps, serving)


@torch.compiler.disable
def scaled_activation(x, groups, inject=False, serving=False):
    return _Activation.apply(x, groups, inject, serving)


@torch.compiler.disable
def sigmoid_mean(x, up, groups, serving=False):
    return _Mix.apply(x, up, groups, serving)


@torch.compiler.disable
def residual_inject(x, h, weights, serving=False):
    """Serving mode consumes raw injection logits; HF mode consumes weights."""
    return _Inject.apply(x, h, weights, serving)


@torch.compiler.disable
def residual_inject_norm(x, h, weights, gain, eps, serving=False):
    return _InjectNorm.apply(x, h, weights, gain, eps, serving)


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, previous_weights, gain, down, up, block, groups, eps, serving):
        if any(w is not None and w.requires_grad for w in (gain, down, up, block)):
            raise ValueError("Manual hyper-connection backward requires frozen projection/norm weights")
        low_precision_inputs("manual HC", x, gain, down, up, *((block,) if block is not None else ()))
        execution_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        if execution_dtype != x.dtype:
            raise TypeError("manual HC: autocast execution dtype must match activations")
        if not torch.is_autocast_enabled(x.device.type) and any(
            w is not None and w.dtype != x.dtype for w in (down, up, block)
        ):
            raise TypeError("manual HC: mixed projection dtypes require autocast")
        if h is None:
            y = x
            normed, r = kernels.norm(y, gain, groups, eps, serving)
        else:
            y, normed, r = kernels.inject_norm(x, h, previous_weights, gain, eps, serving)
        low = torch.nn.functional.linear(normed, down)
        activated = kernels.activation(low, groups, False, serving)
        logits = torch.nn.functional.linear(activated, up)
        mixed = kernels.mix(normed, logits, groups, serving)
        if block is None:
            injection = None
            weights = x.new_empty(0)
        else:
            injection = torch.nn.functional.linear(normed, block)
            weights = injection if serving else kernels.activation(injection, groups, True)
        ctx.save_for_backward(y, normed, r, low, logits, injection,
                              gain, down, up, block, h, previous_weights)
        ctx.groups, ctx.serving = groups, serving
        return mixed, y, weights

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient, residual_gradient, weights_gradient):
        y, normed, r, low, logits, injection, gain, down, up, block, h, previous_weights = ctx.saved_tensors
        groups, serving = ctx.groups, ctx.serving
        gmix, glogits = kernels.mix_backward(normed, logits, gradient, groups, serving)
        with torch.autocast(y.device.type, enabled=False):
            glow = (glogits.reshape(-1, up.shape[0]) @ up.to(logits.dtype)).view_as(low)
        glow = kernels.activation_backward(low, glow, groups, False, serving)
        with torch.autocast(y.device.type, enabled=False):
            gdown = (glow.reshape(-1, down.shape[0]) @ down.to(low.dtype)).view_as(y)
        gblock = None
        if block is not None:
            gw = weights_gradient if serving else kernels.activation_backward(injection, weights_gradient, groups, True)
            with torch.autocast(y.device.type, enabled=False):
                gblock = (gw.reshape(-1, block.shape[0]) @ block.to(injection.dtype)).view_as(y)
        dx = kernels.norm_branches_backward(
            y, gain, r, gmix, gdown, gblock, residual_gradient, groups
        )
        dh = dw = None
        if h is not None:
            dx, dh, dw = kernels.inject_backward(dx, h, previous_weights, serving)
        return dx, dh, dw, None, None, None, None, None, None, None


@torch.compiler.disable
def gated_residual(module, x, h=None, weights=None):
    return _GatedResidual.apply(
        x, h, weights, module.hc_norm.weight, module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
        module.block_inject_weight.weight if module.block_inject_weight is not None else None,
        module.hc_count, module.hc_norm.eps, module.serving,
    )


class TritonGatedResidual(qwen.Qwen4ExpTextGatedResidual):
    def __init__(self, config, use_combine=True, *, serving=False):
        super().__init__(config, use_combine=use_combine)
        self.serving = serving

    def forward(self, hyper_input, normed=None):
        floating_inputs("TritonGatedResidual", hyper_input, self.hc_norm.weight, cuda=False)
        projections = (self.input_mix_weight_down.weight, self.input_mix_weight_up.weight)
        if self.block_inject_weight is not None:
            projections += (self.block_inject_weight.weight,)
        linear_inputs("TritonGatedResidual", hyper_input, *projections)
        if not hyper_input.is_cuda:
            if self.serving:
                raise ValueError("TritonGatedResidual serving: requires CUDA tensors")
            if normed is not None:
                raise ValueError("TritonGatedResidual: precomputed CPU norm is unsupported")
            return super().forward(hyper_input)
        if torch.is_autocast_enabled(hyper_input.device.type) and torch.get_autocast_dtype(hyper_input.device.type) != hyper_input.dtype:
            raise TypeError("TritonGatedResidual: autocast execution dtype must match activations")
        serving = self.serving
        if normed is None:
            normed = grouped_norm(hyper_input, self.hc_norm.weight, self.hc_count, self.hc_norm.eps, serving)
        low = scaled_activation(self.input_mix_weight_down(normed), self.hc_count, serving=serving)
        mixed = sigmoid_mean(normed, self.input_mix_weight_up(low), self.hc_count, serving)
        if self.block_inject_weight is None:
            return mixed
        logits = self.block_inject_weight(normed)
        weights = logits if serving else scaled_activation(logits, self.hc_count, inject=True)
        return mixed, hyper_input, weights


class ManualGatedResidual(TritonGatedResidual):
    def forward(self, hyper_input, injection=None):
        if not hyper_input.is_cuda:
            if injection is not None:
                raise ValueError("ManualGatedResidual: CPU injection is unsupported")
            return super().forward(hyper_input)
        h, weights = injection if injection is not None else (None, None)
        output = gated_residual(self, hyper_input, h, weights)
        return output if self.block_inject_weight is not None else output[0]
