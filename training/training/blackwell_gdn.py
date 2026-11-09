"""Constrain GDN autotuners on the pinned Blackwell stack.

The chunk-state kernel keeps only its two-warp configurations (upstream FLA
#945 / Triton #9871 report a four-warp recurrence race). For the learner's FLA
package, the chunk output kernel also drops its eight-warp BK=BV=128
configuration: on B200 with FLA 0.5.1 and Triton 3.7.1 it returns
nondeterministic, wrong outputs (about 40% relative error against the exact
recurrence from token 32 of each chunk on), and the autotuner selects it.
Tensor layouts, precision, block sizes of the kept configurations, pipeline
stages and all other kernels retain their existing implementations.
"""

from __future__ import annotations

import importlib
from importlib import metadata
import logging

import torch

from training.vllm_runtime import is_flash_vllm_version

log = logging.getLogger(__name__)

_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
_VLLM_MODULES = {
    "0.25.1": "vllm.model_executor.layers.fla.ops.chunk_delta_h",
    "0.29.0": "vllm.third_party.flash_linear_attention.ops.chunk_delta_h",
}
_SUPPORTED_TRITON = {"3.6.0", "3.7.1"}
_MARKER = "_arc3_blackwell_two_warp_state_kernel"
_OUTPUT_KERNEL = "chunk_fwd_kernel_o"
_FLA_OUTPUT_MODULE = "fla.ops.common.chunk_o"
_OUTPUT_MARKER = "_arc3_blackwell_output_kernel_without_eight_warps"
# FLA 0.5.1's complete output-kernel space on non-Hopper GPUs: (BK, BV, warps, stages).
_OUTPUT_SPACE = {(128, 128, 8, 3), (64, 64, 4, 3), (32, 32, 2, 3)}


def _version(name: str, expected: set[str]) -> str:
    try:
        found = metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Blackwell GDN guard requires {name}") from error
    if found not in expected:
        raise RuntimeError(
            f"Blackwell GDN guard supports {name} {sorted(expected)}, found {found}"
        )
    return found


def _restrict_tuner(tuner, *, source: str) -> int:
    """Run before first compilation; reject cached or unfamiliar implementations."""
    tuner = _autotuner(tuner, source=source)
    configs = tuner.configs
    if not configs or any(
        set(config.kwargs) != {"BV"}
        or config.kwargs["BV"] not in (32, 64)
        or config.num_warps not in (2, 4)
        or config.num_stages not in (1, 2, 3, 4)
        for config in configs
    ):
        raise RuntimeError(f"Unexpected Blackwell GDN config space in {source}")
    if getattr(tuner, _MARKER, False):
        best = getattr(tuner, "best_config", None)
        if (
            any(config.num_warps != 2 for config in configs)
            or any(config.num_warps != 2 for config in tuner.cache.values())
            or (best is not None and best.num_warps != 2)
        ):
            raise RuntimeError(f"Blackwell GDN guard was modified in {source}")
        return len(configs)
    if tuner.cache or getattr(tuner, "best_config", None) is not None:
        raise RuntimeError(
            f"Blackwell GDN guard must run before the first kernel invocation: {source}"
        )
    safe = [config for config in configs if config.num_warps == 2]
    if not safe:
        raise RuntimeError(f"No two-warp Blackwell GDN configurations in {source}")
    # Keep the original Config objects so all BV/stage/resource settings survive.
    tuner.configs = safe
    # Triton3.6.0/3.7.1 includes every candidate Config in the disk-cache key.
    # The restricted space gets a distinct key, so retain normal disk caching.
    setattr(tuner, _MARKER, True)
    return len(safe)


def _autotuner(tuner, *, source: str):
    from triton.runtime.autotuner import Autotuner

    seen = set()
    while not isinstance(tuner, Autotuner):
        if tuner is None or id(tuner) in seen:
            raise RuntimeError(f"Blackwell GDN autotuner not found in {source}")
        seen.add(id(tuner))
        tuner = getattr(tuner, "fn", None)
    if not isinstance(tuner.cache, dict) or not isinstance(tuner.configs, list):
        raise RuntimeError(f"Unexpected Blackwell GDN autotuner state in {source}")
    return tuner


def _check_output_tuner(tuner, *, source: str):
    """Validate the output kernel before any tuner is changed."""
    tuner = _autotuner(tuner, source=source)
    if getattr(tuner, _OUTPUT_MARKER, False):
        best = getattr(tuner, "best_config", None)
        if (
            any(config.num_warps > 4 for config in tuner.configs)
            or any(config.num_warps > 4 for config in tuner.cache.values())
            or (best is not None and best.num_warps > 4)
        ):
            raise RuntimeError(f"Blackwell GDN output guard was modified in {source}")
        return tuner
    space = {
        (config.kwargs.get("BK"), config.kwargs.get("BV"), config.num_warps, config.num_stages)
        for config in tuner.configs
    }
    if (
        space != _OUTPUT_SPACE
        or len(tuner.configs) != len(_OUTPUT_SPACE)
        or any(set(config.kwargs) != {"BK", "BV"} for config in tuner.configs)
    ):
        raise RuntimeError(f"Unexpected Blackwell GDN output config space in {source}")
    if tuner.cache or getattr(tuner, "best_config", None) is not None:
        raise RuntimeError(
            f"Blackwell GDN guard must run before the first kernel invocation: {source}"
        )
    return tuner


def _restrict_output_tuner(tuner) -> int:
    if not getattr(tuner, _OUTPUT_MARKER, False):
        # The kept Config objects are the original ones; the disk-cache key
        # changes with the candidate list, so old eight-warp picks never load.
        tuner.configs = [config for config in tuner.configs if config.num_warps <= 4]
        setattr(tuner, _OUTPUT_MARKER, True)
    return len(tuner.configs)


def configure_blackwell_gdn(backend: str, *, device=None) -> bool:
    """Install once per worker process before model warmup or any policy forward."""
    if backend not in ("fla", "vllm"):
        raise ValueError(f"Unknown GDN backend: {backend}")
    if device is not None and torch.device(device).type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        return False
    if backend == "fla":
        # Preserve the existing torch fallback when the optional FLA is absent.
        try:
            metadata.version("fla-core")
        except metadata.PackageNotFoundError:
            return False
        package_version = _version("fla-core", {"0.5.1"})
        _version("flash-linear-attention", {"0.5.1"})
        module_name = "fla.ops.common.chunk_delta_h"
    else:
        package_version = metadata.version("vllm")
        if is_flash_vllm_version(package_version):
            module_name = _VLLM_MODULES["0.29.0"]
        else:
            package_version = _version("vllm", set(_VLLM_MODULES))
            module_name = _VLLM_MODULES[package_version]
    triton_version = _version("triton", _SUPPORTED_TRITON)
    output_tuner = None
    try:
        module = importlib.import_module(module_name)
        tuner = getattr(module, _KERNEL)
        if backend == "fla":
            cache = importlib.import_module("fla.ops.utils.cache")
            # Do not alter process-wide FLA settings. A caller that injects
            # external configs could otherwise override this safe config space.
            if cache.FLA_CACHE_MODE is not cache.FlaCacheMode.DISABLED:
                raise RuntimeError(
                    "Blackwell GDN guard requires FLA_CACHE_MODE=disabled; "
                    "external tuning configs may restore unsafe warp counts"
                )
            output_tuner = getattr(
                importlib.import_module(_FLA_OUTPUT_MODULE), _OUTPUT_KERNEL
            )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"Pinned Blackwell GDN kernel contract changed: {module_name}.{_KERNEL}"
            + (f" / {_FLA_OUTPUT_MODULE}.{_OUTPUT_KERNEL}" if backend == "fla" else "")
        ) from error
    if output_tuner is not None:
        # Validate both kernels before changing either.
        output_tuner = _check_output_tuner(output_tuner, source=_FLA_OUTPUT_MODULE)
    count = _restrict_tuner(tuner, source=module_name)
    output_count = _restrict_output_tuner(output_tuner) if output_tuner is not None else None
    log.info(
        "Blackwell GDN guard: backend=%s version=%s triton=%s capability=%d.%d "
        "kernel=%s num_warps=2 safe_configs=%d; %s",
        backend,
        package_version,
        triton_version,
        *capability,
        _KERNEL,
        count,
        "other kernels unchanged"
        if output_count is None
        else f"{_OUTPUT_KERNEL} without eight warps safe_configs={output_count}; "
        "other kernels unchanged",
    )
    return True
