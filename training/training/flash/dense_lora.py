"""Dense Flash adapters using the target names in training.config.LoraSettings."""

import math
import re
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from .gated_pointwise import scaled_add
from .lora_targets import ALL_TARGETS, DENSE_TARGETS, OPTIONAL_TARGETS, validate_target_modules
from .policies import TrainabilityPolicy
from .dtype_contract import floating_inputs, linear_inputs

# layers.<i>.<DENSE_TARGETS parent>.<target>.lora_<A|B>
_ADAPTER_PARAMETER = re.compile(
    r"layers\.\d+\.(" + "|".join(map(re.escape, DENSE_TARGETS)) + r")\.([^.]+)\.lora_([AB])"
)


@contextmanager
def adapters_disabled(model):
    """Temporarily bypass dense adapter deltas, including nested contexts."""
    modules = [(m, m.adapters_enabled) for m in model.modules() if isinstance(m, LoRALinear)]
    try:
        for module, _ in modules:
            module.adapters_enabled = False
        yield
    finally:
        for module, enabled in modules:
            module.adapters_enabled = enabled


def to_peft_state_dict(trainables, config):
    """Export replicated FP32 masters without wrapper-dependent module names."""
    targets = validate_target_modules(config.target_modules)
    result = {}
    for name, value in trainables.items():
        match = _ADAPTER_PARAMETER.fullmatch(name)
        if match is None or match[2] not in targets:
            raise ValueError(f"Invalid Flash adapter parameter: {name}")
        if value.dtype != torch.float32 or value.ndim != 2:
            raise ValueError(f"Flash adapter masters must be FP32 matrices: {name}")
        if value.shape[0 if match[3] == "A" else 1] != config.r:
            raise ValueError(f"Flash adapter rank mismatch: {name}")
        result[f"base_model.model.model.{name}.weight"] = value.detach().cpu().contiguous().clone()
    if not result:
        raise ValueError("No Flash adapters to export")
    return result


def from_peft_state_dict(state, trainables, config):
    """Validate the complete PEFT census before copying into live parameters."""
    expected = to_peft_state_dict(trainables, config)
    if set(state) != set(expected):
        raise ValueError("Flash PEFT adapter keys do not match the trainable parameters")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != torch.float32:
            raise ValueError(f"Flash PEFT shape/dtype mismatch: {key}")
    with torch.no_grad():
        for name, parameter in trainables.items():
            parameter.copy_(state[f"base_model.model.model.{name}.weight"])



class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, *, fuse_epilogue=False):
        super().__init__()
        if type(r) is not int or r <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Dense LoRA requires positive r and finite positive alpha")
        self.base = base.requires_grad_(False)
        self.scale = alpha / r
        self.fuse_epilogue = fuse_epilogue
        self.adapters_enabled = True
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32)
        )

    def forward(self, x):
        floating_inputs("LoRALinear", x, self.base.weight, self.lora_A, self.lora_B, cuda=False)
        linear_inputs("LoRALinear", x, self.base.weight, *((self.base.bias,) if self.base.bias is not None else ()))
        if not self.adapters_enabled:
            return self.base(x)
        low = F.linear(x, self.lora_A.to(x.dtype))
        if self.fuse_epilogue and x.is_cuda:
            return scaled_add(self.base(x), F.linear(low, self.lora_B.to(x.dtype)), self.scale)
        return self.base(x) + F.linear(low, self.lora_B.to(x.dtype)) * self.scale


class _DenseLoRAConfigurator:
    """Pre-FSDP layer callback that wraps dense targets; routed experts remain frozen.

    Pass ``settings.r``, ``settings.alpha``, and ``settings.target_modules``
    from LoraSettings. An empty target list is rejected to avoid silent no-op
    training. Dropout is not part of this dense learner adapter. Layers lacking
    a target are skipped; ``validate_trainables`` rejects targets matched by no layer.
    """

    def __init__(self, r=16, alpha=32, target_modules=ALL_TARGETS, *, fuse_epilogue=False, group_ranks=None):
        self.targets = validate_target_modules(target_modules)
        if type(r) is not int or r <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Dense LoRA requires positive r and finite positive alpha")
        self.r, self.alpha, self.fuse_epilogue = r, alpha, fuse_epilogue
        # Opt-in: {DENSE_TARGETS parent: (r, alpha)} for that parent's targets (absent: r, alpha everywhere).
        self.group_ranks = {}
        for parent, value in (group_ranks or {}).items():
            if parent not in DENSE_TARGETS:
                raise ValueError(f"Dense LoRA group_ranks: unknown group {parent!r}; allowed {sorted(DENSE_TARGETS)}")
            if (not isinstance(value, (tuple, list)) or len(value) != 2 or type(value[0]) is not int
                    or value[0] <= 0 or not math.isfinite(value[1]) or value[1] <= 0):
                raise ValueError(f"Dense LoRA group_ranks[{parent!r}] must be (positive int r, positive alpha)")
            self.group_ranks[parent] = (value[0], float(value[1]))

    def rank_of(self, parent_name):
        """(r, alpha) of a DENSE_TARGETS parent's adapters."""
        return self.group_ranks.get(parent_name, (self.r, self.alpha))

    def __call__(self, layer, index):
        trainable = {}
        for parent_name, leaves in DENSE_TARGETS.items():
            try:
                parent = layer.get_submodule(parent_name)
            except AttributeError:
                continue
            for leaf in (*leaves, *OPTIONAL_TARGETS.get(parent_name, ())):
                if leaf not in self.targets:
                    continue
                # Leaf names are runtime strings from the configurable target_modules.
                base = getattr(parent, leaf)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Dense LoRA target must be Linear: {parent_name}.{leaf}")
                r, alpha = self.rank_of(parent_name)
                wrapped = LoRALinear(base, r, alpha, fuse_epilogue=self.fuse_epilogue)
                setattr(parent, leaf, wrapped)
                trainable[f"{parent_name}.{leaf}.lora_A"] = wrapped.lora_A
                trainable[f"{parent_name}.{leaf}.lora_B"] = wrapped.lora_B
        return trainable

    def validate_trainables(self, trainable):
        missing = tuple(target for target in self.targets if not any(
            f".{target}.lora_A" in name for name in trainable
        ))
        if missing:
            raise ValueError(f"Dense LoRA: target_modules={self.targets!r}; targets matched nowhere: {missing!r}")


def dense_lora_policy(r=16, alpha=32, target_modules=ALL_TARGETS, *, fuse_epilogue=False, train_routers=False,
                      group_ranks=None, expert_lora=None):
    """Custom trainability that wraps dense targets and checks the model-wide census.

    To customize layer setup (for example, initialization), wrap
    ``policy.configure_layer`` and keep ``validate_trainables`` with
    ``dataclasses.replace(policy, configure_layer=...)``. ``train_routers``
    also trains the FP32 routers, as the separate ``routers`` group.
    Opt-in ``group_ranks`` ({group: (r, alpha)}, e.g. ``{"mlp.shared_expert": (256, 512)}``) gives a
    DENSE_TARGETS group its own rank; opt-in ``expert_lora`` adds routed-expert LoRA (expert_lora.py) as the
    ``expert_lora`` group beside the dense adapters.
    """
    configure = _DenseLoRAConfigurator(r, alpha, target_modules, fuse_epilogue=fuse_epilogue,
                                       group_ranks=group_ranks)
    return TrainabilityPolicy(
        mode="custom", configure_layer=configure, validate_trainables=configure.validate_trainables,
        train_routers=train_routers, expert_lora=expert_lora,
    )
