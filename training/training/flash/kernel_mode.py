"""One switch for the Flash learner's execution stack: ``reference`` or ``fast``.

``reference`` is the stack the cached-teacher distillation results were
produced with: the builder's own defaults (SDPA attention unless a config
selects FlexQSA, native convolutions, the batched FP32 selector, the weighted
expert permutation, the fused norm tail, full layer checkpoints) and FLA's
autotuner left as it is. ``fast`` is the RL learner's stack
(training/FLASH_RL.md, "Model, dtypes and kernels"): group-sparse QSA, FLA
convolutions, the compiled PLE convolution and selector, Triton hyper-connection
and pointwise fusions, the fused expert permutation, small-GEMM selective
checkpoints, replicated frozen routers and the Blackwell GDN guard.

Both pipelines resolve their configuration here, the distillation learner
through its config loader (sft/distillation/config.py) and the RL learner through
``training.flash_config.FlashSettings``, and hand ``builder_kwargs`` of the
result to ``training.flash.model.build_flash_model``. A field set explicitly
overrides the preset's choice; None means the preset's. Placement and memory
knobs (CPU offload, selection reuse and retention, FSDP layout) are not part of
the switch: each pipeline keeps its own.

CPU-light: importable by config loaders and launchers without Torch.
"""

from .options import (
    EXPERT_OPTIMIZATIONS, GDN_CONVOLUTIONS, HYPER_CONNECTIONS, PLE_CONVOLUTIONS, POINTWISE_ROUNDINGS,
    SELECTOR_POLICIES, FlashOptions,
)
from .qsa_policy import validate_qsa_kernel

KERNEL_MODES = ("reference", "fast")

# None for qsa_kernel means the attention backend's kernel under that mode
# (resolve_kernel_stack).
PRESETS = {
    "reference": {
        "attention_backend": "sdpa",
        "qsa_kernel": None,
        "gdn_convolution": "native",
        "ple_convolution": "native",
        "expert_optimization": "weighted_permutation",
        "selector_policy": "batched_fp32_top513_v1",
        "hyper_connection": "none",
        "fuse_norm_tail": True,
        "fuse_gdn_norm": False,
        "fuse_mlp_gates": False,
        "fuse_gdn_qkv": False,
        "fuse_qsa_norm": False,
        "pointwise_rounding": "hf",
        "checkpoint_policy": "full",
        "compile_layers": False,
        "compile_precision": "reference",
        "replicate_frozen_routers": False,
        "gdn_guard": False,
    },
    "fast": {
        "attention_backend": "flex_qsa",
        "qsa_kernel": None,
        "gdn_convolution": "fla",
        "ple_convolution": "compiled",
        "expert_optimization": "fused_permutation",
        "selector_policy": "compiled_fp32_top513_v1",
        "hyper_connection": "triton_manual",
        "fuse_norm_tail": True,
        "fuse_gdn_norm": True,
        "fuse_mlp_gates": True,
        "fuse_gdn_qkv": True,
        "fuse_qsa_norm": True,
        "pointwise_rounding": "hf",
        "checkpoint_policy": "small_gemm",
        "compile_layers": False,
        "compile_precision": "reference",
        "replicate_frozen_routers": True,
        "gdn_guard": True,
    },
}
KERNEL_FIELDS = tuple(PRESETS["reference"])
# The one stack field that is not a build_flash_model keyword argument: the
# caller installs the guard (training/blackwell_gdn.py) before any kernel runs.
NON_BUILDER_FIELDS = ("gdn_guard",)

_CHOICES = {
    "attention_backend": ("sdpa", "flex_qsa"),
    "gdn_convolution": GDN_CONVOLUTIONS,
    "ple_convolution": PLE_CONVOLUTIONS,
    "expert_optimization": EXPERT_OPTIMIZATIONS,
    "selector_policy": SELECTOR_POLICIES,
    "hyper_connection": HYPER_CONNECTIONS,
    "pointwise_rounding": POINTWISE_ROUNDINGS,
    "checkpoint_policy": ("full", "small_gemm"),
    "compile_precision": ("reference", "fused"),
}
_FLAGS = tuple(name for name, value in PRESETS["reference"].items() if type(value) is bool)


def resolve_kernel_stack(mode="reference", overrides=None, *, packed=False, fsdp=True):
    """The complete execution stack: ``mode``'s preset, then explicit overrides.

    ``packed``: the caller feeds packed documents (the RL learner). FLA's
    unguarded autotuner can pick a chunk-output configuration that is wrong on
    the packed (cu_seqlens) GDN path on Blackwell, so a packed caller always
    gets the guard, and the single-document ``physical_sparse`` kernel is not
    available to it. ``fsdp``: the model is FSDP-wrapped, as every distributed
    build is; a single-process build replicates every router already, so
    ``replicate_frozen_routers`` resolves off.
    """
    if mode not in KERNEL_MODES:
        raise ValueError(f"kernel_mode={mode!r}; allowed values are {KERNEL_MODES}")
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(KERNEL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown kernel stack fields: {sorted(unknown)}")
    explicit = {name: value for name, value in overrides.items() if value is not None}
    stack = PRESETS[mode] | explicit
    for name, allowed in _CHOICES.items():
        if stack[name] not in allowed:
            raise ValueError(f"{name}={stack[name]!r}; allowed values are {allowed}")
    for name in _FLAGS:
        if type(stack[name]) is not bool:
            raise ValueError(f"{name}={stack[name]!r}; allowed values are False, True (bool)")
    if stack["qsa_kernel"] is None:
        if stack["attention_backend"] != "flex_qsa":
            stack["qsa_kernel"] = "flex"
        elif mode == "fast":
            stack["qsa_kernel"] = "group_sparse"
        else:
            # physical_sparse is the distillation default for one unpacked
            # document; Flex is the packed-capable kernel it was qualified against.
            stack["qsa_kernel"] = "flex" if packed else "physical_sparse"
    stack["qsa_kernel"] = validate_qsa_kernel(stack["qsa_kernel"], stack["attention_backend"], packed=packed)
    if not fsdp and stack["replicate_frozen_routers"]:
        if explicit.get("replicate_frozen_routers"):
            raise ValueError("replicate_frozen_routers=True requires an FSDP model")
        stack["replicate_frozen_routers"] = False
    if packed and not stack["gdn_guard"]:
        if explicit.get("gdn_guard") is False:
            raise ValueError("gdn_guard=False is unsafe with packed documents (training/blackwell_gdn.py)")
        stack["gdn_guard"] = True
    # FlashOptions checks the cross-field rules of the module options.
    FlashOptions(**{name: stack[name] for name in (
        "gdn_convolution", "ple_convolution", "selector_policy", "qsa_kernel", "expert_optimization",
        "hyper_connection", "fuse_gdn_norm", "fuse_mlp_gates", "fuse_gdn_qkv", "fuse_qsa_norm",
        "pointwise_rounding",
    )})
    return stack


def builder_kwargs(stack):
    """The ``build_flash_model`` keyword arguments of a resolved stack."""
    return {name: value for name, value in stack.items() if name not in NON_BUILDER_FIELDS}
