"""Construction-time execution choices for Flash adapters.

These are internal module options, not an additional checkpoint-manifest policy.
Validation is shared by direct construction and the public builder.
"""

from dataclasses import dataclass

from .qsa_policy import QSA_KERNELS


SELECTOR_POLICIES = (
    "batched_fp32_top513_v1",
    "compiled_fp32_top513_v1",
    "bf16_dot_fp32_top513_v1",
)
EXPERT_OPTIMIZATIONS = ("none", "no_sentinel", "weighted_permutation", "fused_permutation")
GDN_CONVOLUTIONS = ("native", "fla")
PLE_CONVOLUTIONS = ("native", "compiled")
HYPER_CONNECTIONS = ("none", "triton", "triton_inject_norm", "triton_manual")
POINTWISE_ROUNDINGS = ("hf", "vllm")


@dataclass(frozen=True)
class FlashOptions:
    gdn_convolution: str = "native"
    ple_convolution: str = "native"
    selector_policy: str = "batched_fp32_top513_v1"
    qsa_kernel: str | None = None
    expert_optimization: str = "none"
    hyper_connection: str = "none"
    gdn_grouped_heads: bool = False
    fuse_gdn_norm: bool = False
    fuse_mlp_gates: bool = False
    fuse_gdn_qkv: bool = False
    fuse_qsa_norm: bool = False
    pointwise_rounding: str = "hf"

    def __post_init__(self):
        for name, allowed in (
            ("gdn_convolution", GDN_CONVOLUTIONS),
            ("ple_convolution", PLE_CONVOLUTIONS),
            ("selector_policy", SELECTOR_POLICIES),
            ("qsa_kernel", (None, *QSA_KERNELS)),
            ("expert_optimization", EXPERT_OPTIMIZATIONS),
            ("hyper_connection", HYPER_CONNECTIONS),
            ("pointwise_rounding", POINTWISE_ROUNDINGS),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"FlashOptions: {name}={value!r}; allowed values are {allowed}")
        for name in ("gdn_grouped_heads", "fuse_gdn_norm", "fuse_mlp_gates", "fuse_gdn_qkv", "fuse_qsa_norm"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(f"FlashOptions: {name}={value!r}; allowed values are False, True (bool)")
        if self.gdn_grouped_heads:
            raise ValueError("FlashOptions: gdn_grouped_heads=True is unqualified; see tests/distillation/test_gdn_grouped_heads.py")
        if self.pointwise_rounding == "vllm" and (not self.fuse_gdn_norm or self.hyper_connection == "none"):
            raise ValueError(
                "FlashOptions: pointwise_rounding='vllm' requires fuse_gdn_norm=True and "
                "hyper_connection in ('triton', 'triton_inject_norm', 'triton_manual') for "
                "GDN gated norm and HC norm, activation, mixing and injection; "
                "see tests/distillation/test_serving_rounding.py"
            )
