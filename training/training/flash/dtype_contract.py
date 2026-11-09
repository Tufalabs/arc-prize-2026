"""Python-boundary contracts for the Flash fused kernels."""

import torch

FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def floating_inputs(op, *inputs, cuda=True, same_dtype=False):
    for i, value in enumerate(inputs):
        if value.dtype not in FLOAT_DTYPES:
            raise TypeError(f"{op}: unsupported input {i} dtype {value.dtype}")
        if cuda and not value.is_cuda:
            raise ValueError(f"{op}: requires CUDA tensors, got {value.device}")
        if value.device != inputs[0].device:
            raise ValueError(f"{op}: inputs must share a device")
        if same_dtype and value.dtype != inputs[0].dtype:
            raise TypeError(f"{op}: inputs must share a dtype, got {inputs[0].dtype} and {value.dtype}")


def low_precision_inputs(op, *inputs, same_dtype=False):
    floating_inputs(op, *inputs, same_dtype=same_dtype)
    if inputs[0].dtype == torch.float32:
        raise TypeError(f"{op}: FP32 activations are unsupported")


def index_inputs(op, activation, *indices):
    for value in indices:
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{op}: indices must be int32/int64, got {value.dtype}")
        if value.device != activation.device:
            raise ValueError(f"{op}: indices must be on {activation.device}, got {value.device}")


def linear_inputs(op, activation, *weights):
    floating_inputs(op, activation, *weights, cuda=False)
    if not torch.is_autocast_enabled(activation.device.type) and any(
        weight.dtype != activation.dtype for weight in weights
    ):
        raise TypeError(f"{op}: mixed projection dtypes require autocast")
