"""Flash dense LoRA target names; safe for config and launcher processes."""

DENSE_TARGETS = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"),
    "mlp.shared_expert": ("gate_proj", "up_proj", "down_proj"),
}
ALL_TARGETS = tuple(name for names in DENSE_TARGETS.values() for name in names)
OPTIONAL_TARGETS = {"linear_attn": ("in_proj_b", "in_proj_a")}


PEFT_VERSION = "0.18.1"


def peft_adapter_config(*, base_model_name_or_path, r, alpha, target_modules):
    """The adapter_config.json the trainer publishes and the PEFT helpers write."""
    return {
        "base_model_name_or_path": str(base_model_name_or_path),
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": int(r), "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": list(validate_target_modules(target_modules)),
        "use_rslora": False, "use_dora": False, "modules_to_save": None,
        "fan_in_fan_out": False, "inference_mode": False, "init_lora_weights": True,
        "rank_pattern": {}, "alpha_pattern": {}, "peft_version": PEFT_VERSION,
    }


def validate_target_modules(target_modules):
    allowed = {*ALL_TARGETS, *OPTIONAL_TARGETS["linear_attn"]}
    if (not isinstance(target_modules, (list, tuple)) or not target_modules
            or any(not isinstance(name, str) or name not in allowed for name in target_modules)
            or len(set(target_modules)) != len(target_modules)):
        raise ValueError(f"Unknown, duplicate or empty Flash LoRA target_modules: {target_modules!r}")
    return tuple(target_modules)
