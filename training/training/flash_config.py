"""CPU-safe configuration for the Flash learner."""

from dataclasses import asdict, dataclass, field, fields, replace
import logging
import math

from training.flash.kernel_mode import KERNEL_FIELDS, builder_kwargs, resolve_kernel_stack
from training.flash_policy_weights import ROUNDINGS

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ParityThresholds:
    """Same-checkpoint trainer/sampler logprob parity tolerances (training/ppo.py).

    The class defaults are the LoRA/full-weight tolerances; Flash uses flash().
    """

    mean_abs: float = 0.05
    p95_abs: float = 0.25
    outlier_abs: float = 2.0
    max_outlier_fraction: float = 0.001
    marginal_band_factor: float = 1.5

    @classmethod
    def flash(cls):
        """Flash defaults above the vLLM fork's own floor on e307.

        The learner sits at the serving engine's self-consistency floor, which
        depends on the kind of request checked: decode against the engine's
        own prefill differs by mean |delta logprob| 0.05 on initial prompts
        and 0.13-0.15 (p95 about 0.6) on short prune-mode compaction prompts,
        mostly MoE routing flips. At 0.17/0.75 a correct learner failed 9-20%
        of reset-sample checks; at 0.25/1.00, with the gate always expanding
        to 1,000 targets before failing, 0-0.6%. Derivation: training/FLASH_RL.md,
        "Parity thresholds".
        """
        return cls(mean_abs=0.25, p95_abs=1.0, outlier_abs=2.0, max_outlier_fraction=0.01,
                   marginal_band_factor=1.5)

    @classmethod
    def from_dict(cls, payload, defaults=None):
        """Validate an override; keys it omits keep `defaults` (the class defaults if None)."""
        values = _section(cls, payload, "parity_thresholds")
        result = replace(cls() if defaults is None else defaults, **values)
        for name, value in asdict(result).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"parity_thresholds.{name} must be finite and positive")
        if result.max_outlier_fraction > 1 or result.marginal_band_factor < 1:
            raise ValueError("Invalid parity fraction or marginal band")
        return result


def _section(cls, payload, name):
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"training.{name} must be an object")
    unknown = set(payload) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown training.{name} keys: {sorted(unknown)}")
    return payload


# Settings that only change what the learner measures, never what it trains:
# a resume may change them (flash_ppo._recipe_identity).
DIAGNOSTIC_SETTINGS = ("load_balance_metrics", "load_balance_eval_episodes")

# Removed settings that frozen configs and trainer states written before the
# removal still carry: ignored, with one warning per process. Each maps to the
# values it accepted while it existed (None: any); any other value is refused,
# as it always was.
# adapter_probe: the amplified-adapter probe (training/FLASH_RL.md, "Sharp edges
# of LoRA publication").
# expert_storage: routed experts are always BF16 (training/FLASH_RL.md,
# "Expert storage"); NVFP4 storage computed bitwise the same.
RETIRED_SETTINGS = {"adapter_probe": None, "expert_storage": ("nvfp4", "bf16")}
_warned_retired = set()


def warn_retired_setting(name, source):
    """Log once per process that the retired ``training.flash.<name>`` in ``source`` is ignored."""
    if name not in _warned_retired:
        _warned_retired.add(name)
        log.warning("ignoring the retired setting training.flash.%s in %s: it has no effect", name, source)


def drop_retired_settings(payload, source):
    """``payload`` without its retired settings, each warned about once per process.

    A retired setting holding a value it never accepted is refused.
    """
    for name, accepted in RETIRED_SETTINGS.items():
        if name not in payload:
            continue
        if accepted is not None and payload[name] not in accepted:
            raise ValueError(f"training.flash.{name} must be one of {accepted} in {source}; "
                             "the setting is retired and has no effect")
        warn_retired_setting(name, source)
    return {key: value for key, value in payload.items() if key not in RETIRED_SETTINGS}


# The learner's kernel stack fields: the GDN guard is not one of them, since
# the learner packs documents and always installs it (PPOTrainer).
_KERNEL_SETTINGS = tuple(name for name in KERNEL_FIELDS if name != "gdn_guard")


@dataclass(slots=True)
class FlashSettings:
    checkpoint_root: str = ""
    device_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # The execution stack (training/flash/kernel_mode.py): "fast" is the RL
    # stack, "reference" the one the distillation results were produced with.
    # The fields from attention_backend to replicate_frozen_routers override
    # single choices of the preset; None (the default) takes the preset's, and
    # the loaded settings hold the resolved values.
    kernel_mode: str = "fast"
    attention_backend: str | None = None
    # None also follows the backend: group_sparse (fast) or flex (reference)
    # under flex_qsa, the dense Flex mask under SDPA.
    qsa_kernel: str | None = None
    gdn_convolution: str | None = None
    ple_convolution: str | None = None
    expert_optimization: str | None = None
    selector_policy: str | None = None
    hyper_connection: str | None = None
    fuse_norm_tail: bool | None = None
    fuse_gdn_norm: bool | None = None
    fuse_mlp_gates: bool | None = None
    fuse_gdn_qkv: bool | None = None
    fuse_qsa_norm: bool | None = None
    pointwise_rounding: str | None = None
    checkpoint_policy: str | None = None
    # Placement, not part of the switch: offload every Nth layer's checkpoint
    # inputs to the host (0: none).
    offload_every: int = 0
    compile_layers: bool | None = None
    compile_precision: str | None = None
    replicate_frozen_routers: bool | None = None
    # Routers as trainables with their own optimizer group, published to the
    # servers with every adapter (training/FLASH_RL.md, "Trainable routers").
    train_routers: bool = False
    decision_replay: bool = False
    group_learning_rates: dict[str, float] = field(default_factory=dict)
    parity_thresholds: ParityThresholds = field(default_factory=ParityThresholds.flash)
    # Expert load-balance metrics (training/expert_load.py): observation only,
    # never a loss. The eval pass scores up to this many official-eval
    # episodes per window (0 disables it).
    load_balance_metrics: bool = True
    load_balance_eval_episodes: int = 8
    # How each publication rounds its FP32 merge W + alpha/r B A to BF16
    # (training/flash_policy_weights.py, merge_lora_weight): "dither" keeps
    # updates below half a BF16 ulp in expectation, "nearest" is the plain cast.
    # It shapes what the servers sample, not the learner's numerics, so a
    # resume may change it (flash_ppo._recipe_identity).
    policy_weights_rounding: str = "dither"

    def __post_init__(self):
        stack = resolve_kernel_stack(
            self.kernel_mode, {name: getattr(self, name) for name in _KERNEL_SETTINGS}, packed=True,
        )
        for name in _KERNEL_SETTINGS:
            setattr(self, name, stack[name])

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, dict):
            payload = drop_retired_settings(payload, "the config")
        values = dict(_section(cls, payload, "flash"))
        values["parity_thresholds"] = ParityThresholds.from_dict(
            values.get("parity_thresholds"), defaults=ParityThresholds.flash())
        try:
            # Resolves the kernel stack and validates its fields.
            result = cls(**values)
        except ValueError as error:
            raise ValueError(f"training.flash: {error}") from error
        # The learner's packed batches and its LoRA/router trainables rule out
        # the distillation-only single-document kernels and expert paths.
        for name, allowed in (("qsa_kernel", ("flex", "group_sparse")),
                              ("expert_optimization", ("weighted_permutation", "fused_permutation")),
                              ("policy_weights_rounding", ROUNDINGS)):
            if getattr(result, name) not in allowed:
                raise ValueError(f"training.flash.{name} must be one of {allowed}")
        for name in ("train_routers", "decision_replay", "load_balance_metrics"):
            if type(getattr(result, name)) is not bool:
                raise ValueError(f"training.flash.{name} must be boolean")
        if not isinstance(result.checkpoint_root, str):
            raise ValueError("training.flash.checkpoint_root must be a string")
        if type(result.offload_every) is not int or result.offload_every < 0:
            raise ValueError("training.flash.offload_every must be a nonnegative integer")
        if type(result.load_balance_eval_episodes) is not int or result.load_balance_eval_episodes < 0:
            raise ValueError("training.flash.load_balance_eval_episodes must be a nonnegative integer")
        ids = result.device_ids
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("training.flash.device_ids must be distinct nonnegative GPU ids")
        rates = result.group_learning_rates
        if not isinstance(rates, dict) or set(rates) - {"custom", "routers", "expert_lora"}:
            raise ValueError("training.flash.group_learning_rates has unknown parameter groups")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rates.values()):
            raise ValueError("Flash group learning rates must be finite and positive")
        if "routers" in rates and not result.train_routers:
            raise ValueError("training.flash.group_learning_rates.routers requires train_routers")
        if result.decision_replay and result.attention_backend != "flex_qsa":
            raise ValueError("Flash decision_replay requires flex_qsa")
        return result

    def kernel_stack(self):
        """The resolved execution stack, guard included (always on for the learner)."""
        return {name: getattr(self, name) for name in _KERNEL_SETTINGS} | {"gdn_guard": True}

    def model_kwargs(self):
        return builder_kwargs(self.kernel_stack()) | {"offload_every": self.offload_every}
