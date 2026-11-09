"""training.flash.kernel_mode: the reference/fast presets and the RL learner's use of them."""

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from flash_configs import RECIPE, write_arc3_flash_config
from training.flash.kernel_mode import KERNEL_FIELDS, PRESETS, builder_kwargs, resolve_kernel_stack
from training.flash_config import FlashSettings

ROOT = Path(__file__).resolve().parents[1]

# The RL learner's stack before kernel_mode existed (FlashSettings defaults at
# cb6da0ef): "fast" must reproduce it exactly.
RL_STACK_OF_RECORD = {
    "attention_backend": "flex_qsa", "qsa_kernel": "group_sparse", "gdn_convolution": "fla",
    "ple_convolution": "compiled", "expert_optimization": "fused_permutation",
    "selector_policy": "compiled_fp32_top513_v1", "hyper_connection": "triton_manual",
    "fuse_norm_tail": True, "fuse_gdn_norm": True, "fuse_mlp_gates": True, "fuse_gdn_qkv": True,
    "fuse_qsa_norm": True, "pointwise_rounding": "hf", "checkpoint_policy": "small_gemm",
    "offload_every": 0, "compile_layers": False, "compile_precision": "reference",
    "replicate_frozen_routers": True,
}
# Placement settings the switch leaves to each pipeline. (The record's NVFP4
# expert storage is retired: experts are always BF16, bitwise the same.)
PLACEMENT = ("offload_every",)


def _defaults(path, function):
    """Keyword defaults of a function, read from source (the builders need Triton to import)."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    arguments = node.args
    positional = arguments.posonlyargs + arguments.args
    pairs = list(zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults))
    pairs += [(a, d) for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults) if d is not None]
    return {argument.arg: ast.literal_eval(default) for argument, default in pairs
            if not isinstance(default, ast.Name)}


def test_reference_is_the_builders_own_defaults():
    builder = _defaults("training/flash/model.py", "build_flash_model")
    reference = builder_kwargs(PRESETS["reference"])
    assert {name: builder[name] for name in reference} == reference
    # The distillation adapter's historical defaults are the same stack.
    adapter = _defaults("sft/distillation/flash_model.py", "build_model")
    for name in ("attention_backend", "qsa_kernel", "expert_optimization", "fuse_norm_tail"):
        assert adapter[name] == PRESETS["reference"][name]
    assert resolve_kernel_stack("reference")["qsa_kernel"] == "flex"


def test_fast_is_the_rl_stack_of_record():
    fast = resolve_kernel_stack("fast", packed=True)
    assert {name: fast[name] for name in RL_STACK_OF_RECORD if name not in PLACEMENT} == {
        name: value for name, value in RL_STACK_OF_RECORD.items() if name not in PLACEMENT}
    assert fast["gdn_guard"] and set(fast) == set(KERNEL_FIELDS)
    for settings in (FlashSettings(), FlashSettings.from_dict({}), FlashSettings.from_dict({"kernel_mode": "fast"})):
        assert settings.kernel_mode == "fast"
        assert {name: getattr(settings, name) for name in RL_STACK_OF_RECORD} == RL_STACK_OF_RECORD
        assert settings.model_kwargs() == RL_STACK_OF_RECORD


def test_shipped_recipe_builds_the_fast_stack():
    from training.config import ExperimentConfig

    settings = ExperimentConfig.load(RECIPE).training.flash
    assert settings.kernel_mode == "fast"
    assert {name: getattr(settings, name) for name in RL_STACK_OF_RECORD} == RL_STACK_OF_RECORD | {
        "offload_every": 4}  # placement: the recipe's 98k-token micro-batches need offload


@pytest.mark.parametrize("mode,backend,packed,kernel", [
    ("fast", "flex_qsa", True, "group_sparse"), ("fast", "flex_qsa", False, "group_sparse"),
    ("reference", "flex_qsa", False, "physical_sparse"), ("reference", "flex_qsa", True, "flex"),
    ("fast", "sdpa", True, "flex"), ("reference", "sdpa", False, "flex"),
])
def test_qsa_kernel_follows_mode_backend_and_packing(mode, backend, packed, kernel):
    assert resolve_kernel_stack(mode, {"attention_backend": backend}, packed=packed)["qsa_kernel"] == kernel


def test_packed_inputs_always_get_the_gdn_guard():
    assert not resolve_kernel_stack("reference")["gdn_guard"]
    assert resolve_kernel_stack("reference", packed=True)["gdn_guard"]
    assert resolve_kernel_stack("reference", {"gdn_guard": True})["gdn_guard"]
    assert not resolve_kernel_stack("fast", {"gdn_guard": False})["gdn_guard"]
    with pytest.raises(ValueError, match="gdn_guard=False is unsafe"):
        resolve_kernel_stack("fast", {"gdn_guard": False}, packed=True)
    with pytest.raises(ValueError, match="one unpacked document"):
        resolve_kernel_stack("reference", {"attention_backend": "flex_qsa", "qsa_kernel": "physical_sparse"},
                             packed=True)


def test_router_replication_needs_fsdp():
    assert resolve_kernel_stack("fast")["replicate_frozen_routers"]
    assert not resolve_kernel_stack("fast", fsdp=False)["replicate_frozen_routers"]
    with pytest.raises(ValueError, match="requires an FSDP model"):
        resolve_kernel_stack("reference", {"replicate_frozen_routers": True}, fsdp=False)


def test_overrides_change_one_field_and_none_takes_the_preset():
    stack = resolve_kernel_stack("fast", {"checkpoint_policy": "full", "hyper_connection": None})
    assert stack == resolve_kernel_stack("fast") | {"checkpoint_policy": "full"}
    assert set(stack) == set(KERNEL_FIELDS)


@pytest.mark.parametrize("mode,overrides", [
    ("turbo", {}), (None, {}), ("fast", {"typo": 1}), ("fast", {"fuse_gdn_norm": 1}),
    ("reference", {"expert_storage": "fp8"}), ("reference", {"attention_backend": "flash"}),
    ("reference", {"qsa_kernel": "group_sparse"}), ("reference", {"pointwise_rounding": "vllm"}),
    ("fast", {"gdn_convolution": "cuda"}),
])
def test_invalid_stacks(mode, overrides):
    with pytest.raises(ValueError):
        resolve_kernel_stack(mode, overrides)


def test_flash_settings_reference_mode_and_overrides():
    reference = FlashSettings.from_dict({"kernel_mode": "reference"})
    assert reference.model_kwargs() == builder_kwargs(resolve_kernel_stack("reference", packed=True)) | {
        "offload_every": 0}
    assert reference.kernel_stack()["gdn_guard"]
    tuned = FlashSettings.from_dict({"kernel_mode": "reference", "offload_every": 2,
                                     "attention_backend": "flex_qsa"})
    assert (tuned.offload_every, tuned.qsa_kernel, tuned.hyper_connection) == (2, "flex", "none")
    assert FlashSettings.from_dict(asdict(tuned)) == tuned
    for payload in ({"kernel_mode": "turbo"}, {"kernel_mode": "reference", "expert_optimization": "none"},
                    {"kernel_mode": "reference", "attention_backend": "flex_qsa", "qsa_kernel": "physical_sparse"}):
        with pytest.raises(ValueError, match="training.flash"):
            FlashSettings.from_dict(payload)


def test_recipe_identity_records_the_switch():
    pytest.importorskip("triton")
    from training.flash_ppo import _recipe_identity

    fast = asdict(FlashSettings.from_dict({"checkpoint_root": "/c"}))
    legacy = {key: value for key, value in fast.items() if key != "kernel_mode"}
    reference = asdict(FlashSettings.from_dict({"checkpoint_root": "/c", "kernel_mode": "reference"}))
    # States saved before the switch ran the fast stack and resume as fast.
    assert _recipe_identity(legacy) == _recipe_identity(fast) != _recipe_identity(reference)
    relabelled = dict(fast, kernel_mode="reference")
    assert _recipe_identity(relabelled) != _recipe_identity(fast)


@pytest.mark.parametrize("mode", ["fast", "reference"])
def test_the_switch_reaches_the_arc3_flash_config(tmp_path, mode):
    from training.config import ExperimentConfig
    from training.job_shape import job_shape

    path = write_arc3_flash_config(tmp_path, kernel_mode=mode)
    config = ExperimentConfig.load(path)
    assert config.inference.environment == "arc3" and config.training.weight_mode == "flash"
    settings = config.training.flash
    assert settings.kernel_mode == mode
    assert settings.model_kwargs() == builder_kwargs(resolve_kernel_stack(mode, packed=True)) | {
        "offload_every": 4}  # the recipe's placement, which the switch leaves alone
    assert job_shape(config, {})["SERVER_ENVIRONMENT"] == "arc3"
    effective = config.to_dict()["training"]["flash"]
    assert effective["kernel_mode"] == mode and effective["attention_backend"] == settings.attention_backend
