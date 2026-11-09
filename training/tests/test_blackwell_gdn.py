import importlib
import sys
from importlib import metadata
from types import SimpleNamespace

import pytest

from training import blackwell_gdn as guard

triton = pytest.importorskip("triton")
Autotuner = pytest.importorskip("triton.runtime.autotuner").Autotuner


@triton.jit
def noop_kernel_for_disk_cache():
    pass


def make_tuner():
    # Real Config/Autotuner state, without a compiled kernel or CUDA allocation.
    tuner = Autotuner.__new__(Autotuner)
    tuner.configs = [
        triton.Config({"BV": bv}, num_warps=warps, num_stages=stages)
        for warps in (2, 4)
        for bv in (32, 64)
        for stages in (2, 3, 4)
    ]
    tuner.cache = {}
    tuner.cache_results = True
    return tuner


def make_output_tuner():
    # FLA 0.5.1 chunk_fwd_kernel_o on non-Hopper GPUs.
    tuner = Autotuner.__new__(Autotuner)
    tuner.configs = [
        triton.Config({"BK": 128, "BV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BK": 32, "BV": 32}, num_warps=2, num_stages=3),
    ]
    tuner.cache = {}
    tuner.cache_results = True
    return tuner


@pytest.fixture
def runtime(monkeypatch):
    tuner = make_tuner()
    output = make_output_tuner()
    disabled = object()
    cache = SimpleNamespace(
        FLA_CACHE_MODE=disabled, FlaCacheMode=SimpleNamespace(DISABLED=disabled)
    )
    versions = {
        "fla-core": "0.5.1",
        "flash-linear-attention": "0.5.1",
        "vllm": "0.25.1",
        "triton": "3.6.0",
    }
    modules = {
        "fla.ops.common.chunk_delta_h": SimpleNamespace(
            **{guard._KERNEL: SimpleNamespace(fn=tuner)}
        ),
        "fla.ops.utils.cache": cache,
        guard._FLA_OUTPUT_MODULE: SimpleNamespace(
            **{guard._OUTPUT_KERNEL: SimpleNamespace(fn=output)}
        ),
        **{
            name: SimpleNamespace(**{guard._KERNEL: SimpleNamespace(fn=tuner)})
            for name in guard._VLLM_MODULES.values()
        },
    }
    imported = []
    real_import = importlib.import_module

    def fake_import(name, *args):
        if name in modules:
            imported.append(name)
            return modules[name]
        return real_import(name, *args)

    monkeypatch.setattr(guard.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(guard.importlib, "import_module", fake_import)
    monkeypatch.setattr(guard.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        guard.torch.cuda, "get_device_capability", lambda device: (10, 0)
    )
    return SimpleNamespace(
        tuner=tuner, output=output, cache=cache, versions=versions, modules=modules,
        imported=imported,
    )


def test_safe_space_preserves_all_original_block_and_stage_settings(runtime, caplog):
    original = runtime.tuner.configs[:]
    with caplog.at_level("INFO"):
        assert guard.configure_blackwell_gdn("fla", device="cuda:1")
    assert runtime.tuner.configs == [
        config for config in original if config.num_warps == 2
    ]
    assert all(
        any(config is old for old in original) for config in runtime.tuner.configs
    )
    assert {(c.kwargs["BV"], c.num_stages) for c in runtime.tuner.configs} == {
        (bv, stages) for bv in (32, 64) for stages in (2, 3, 4)
    }
    assert runtime.tuner.cache_results is True
    assert "capability=10.0" in caplog.text
    assert "num_warps=2 safe_configs=6" in caplog.text


@pytest.mark.parametrize("version", ["0.25.1", "0.29.0", "0.29.1.dev166+g3bda1ba09.precompiled"])
def test_native_exact_version_selects_corresponding_vendored_kernel(runtime, version):
    runtime.versions["vllm"] = version
    runtime.versions["triton"] = "3.6.0" if version == "0.25.1" else "3.7.1"
    assert guard.configure_blackwell_gdn("vllm")
    assert runtime.imported == [guard._VLLM_MODULES["0.29.0" if guard.is_flash_vllm_version(version) else version]]
    assert all(config.num_warps == 2 for config in runtime.tuner.configs)


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (12, 0)])
def test_other_architectures_do_not_import_or_change_any_kernel(
    runtime, monkeypatch, capability
):
    before = runtime.tuner.configs
    monkeypatch.setattr(
        guard.torch.cuda, "get_device_capability", lambda device: capability
    )
    assert not guard.configure_blackwell_gdn("fla")
    assert runtime.tuner.configs is before
    assert runtime.tuner.cache_results is True
    assert not runtime.imported


def test_cpu_trainer_with_visible_cuda_is_unchanged(runtime):
    assert not guard.configure_blackwell_gdn("fla", device="cpu")
    assert not runtime.imported


def test_no_cuda_does_not_inspect_packages(runtime, monkeypatch):
    monkeypatch.setattr(guard.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(guard.metadata, "version", lambda name: pytest.fail(name))
    assert not guard.configure_blackwell_gdn("vllm")


def test_optional_fla_absence_preserves_existing_torch_fallback(runtime, monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(guard.metadata, "version", missing)
    assert not guard.configure_blackwell_gdn("fla")
    assert not runtime.imported


@pytest.mark.parametrize("name", ["fla-core", "flash-linear-attention", "triton"])
def test_unknown_dependency_versions_fail_before_mutating_tuner(runtime, name):
    runtime.versions[name] = "999.0"
    before = runtime.tuner.configs
    with pytest.raises(RuntimeError, match=f"supports {name}"):
        guard.configure_blackwell_gdn("fla")
    assert runtime.tuner.configs is before


def test_external_fla_cache_is_rejected_without_changing_global_mode(runtime):
    other_mode = object()
    runtime.cache.FLA_CACHE_MODE = other_mode
    before = runtime.tuner.configs
    with pytest.raises(RuntimeError, match="FLA_CACHE_MODE=disabled"):
        guard.configure_blackwell_gdn("fla")
    assert runtime.cache.FLA_CACHE_MODE is other_mode
    assert runtime.tuner.configs is before


@pytest.mark.parametrize("cached", [False, True])
def test_late_installation_is_fatal_and_does_not_clear_evidence(runtime, cached):
    unsafe = next(config for config in runtime.tuner.configs if config.num_warps == 4)
    if cached:
        runtime.tuner.cache["observed"] = unsafe
    else:
        runtime.tuner.best_config = unsafe
    with pytest.raises(RuntimeError, match="before the first kernel invocation"):
        guard.configure_blackwell_gdn("fla")
    assert unsafe in runtime.tuner.configs
    if cached:
        assert runtime.tuner.cache["observed"] is unsafe
    else:
        assert runtime.tuner.best_config is unsafe


def test_repeated_setup_preserves_safe_runtime_tuning_cache(runtime):
    guard.configure_blackwell_gdn("fla")
    cached = runtime.tuner.configs[0]
    runtime.tuner.cache["shape"] = cached
    runtime.tuner.best_config = cached
    assert guard.configure_blackwell_gdn("fla")
    assert runtime.tuner.cache == {"shape": cached}
    assert runtime.tuner.cache_results is True


def test_real_triton_disk_cache_separates_old_space_and_reuses_safe_space(
    runtime, monkeypatch, tmp_path
):
    # Execute the installed Triton's real key generation, serialization, and
    # loading code. Only the GPU/backend identity and cache directory are fake.
    # An old disk result deliberately prefers an unsafe four-warp config.
    from triton.compiler import compiler
    from triton.runtime import autotuner

    observed_keys = []

    class DiskCache:
        def __init__(self, key):
            observed_keys.append(key)
            self.directory = tmp_path / key
            self.directory.mkdir(exist_ok=True)

        def get_file(self, name):
            path = self.directory / name
            return str(path) if path.exists() else None

        def put(self, text, name, binary=False):
            assert binary is False
            (self.directory / name).write_text(text)

    monkeypatch.setattr(autotuner, "get_cache_manager", DiskCache)
    monkeypatch.setattr(autotuner, "get_cache_invalidating_env_vars", lambda: {})
    monkeypatch.setattr(autotuner, "triton_key", lambda: "installed-triton-build")
    monkeypatch.setattr(
        autotuner,
        "driver",
        SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: "same-b200")),
    )
    monkeypatch.setattr(
        compiler, "make_backend", lambda target: SimpleNamespace(hash=lambda: target)
    )

    def populate(tuner):
        tuner.configs_timings = {
            config: [0.0 if config.num_warps == 4 else 1.0, 1.0, 1.0]
            for config in tuner.configs
        }

    key = (48, 128, 128, 64, "torch.bfloat16")
    old = make_tuner()
    old.fn = noop_kernel_for_disk_cache
    assert old.check_disk_cache(key, old.configs, lambda: populate(old)) is False

    guard.configure_blackwell_gdn("fla")
    safe = runtime.tuner
    safe.fn = noop_kernel_for_disk_cache
    assert safe.check_disk_cache(key, safe.configs, lambda: populate(safe)) is False
    assert observed_keys[0] != observed_keys[1]

    # The second safe worker should reuse the disk result without benchmarking.
    assert (
        safe.check_disk_cache(
            key, safe.configs, lambda: pytest.fail("unnecessary retune")
        )
        is True
    )
    assert observed_keys[1] == observed_keys[2]
    assert safe.cache[key].num_warps == 2
    assert safe.cache_results is True


def test_repeated_setup_detects_guard_tampering(runtime):
    guard.configure_blackwell_gdn("fla")
    runtime.tuner.cache["unsafe"] = triton.Config({"BV": 32}, num_warps=4)
    with pytest.raises(RuntimeError, match="guard was modified"):
        guard.configure_blackwell_gdn("fla")


def test_changed_kernel_symbol_is_fatal(runtime):
    delattr(runtime.modules["fla.ops.common.chunk_delta_h"], guard._KERNEL)
    with pytest.raises(RuntimeError, match="kernel contract changed"):
        guard.configure_blackwell_gdn("fla")


def test_changed_kernel_config_space_is_fatal(runtime):
    runtime.tuner.configs.append(triton.Config({"BK": 128}, num_warps=2))
    with pytest.raises(RuntimeError, match="Unexpected Blackwell GDN config space"):
        guard.configure_blackwell_gdn("fla")


def test_missing_safe_config_is_fatal(runtime):
    runtime.tuner.configs = [
        config for config in runtime.tuner.configs if config.num_warps == 4
    ]
    with pytest.raises(RuntimeError, match="No two-warp"):
        guard.configure_blackwell_gdn("fla")


def test_unfamiliar_vllm_version_is_not_patched(runtime):
    runtime.versions["vllm"] = "0.30.0"
    with pytest.raises(RuntimeError, match="supports vllm"):
        guard.configure_blackwell_gdn("vllm")
    assert not runtime.imported


@pytest.mark.parametrize("fails", [False, True])
def test_serving_worker_guards_selected_device_before_load_and_profile(
    monkeypatch, fails
):
    calls = []

    class Worker:
        def init_device(self):
            self.device = "cuda:3"
            calls.append("device")

        def load_model(self):
            calls.append("load")

        def profile_run(self):
            calls.append("profile")

    def configure(backend, device):
        assert backend == "vllm"
        assert device == "cuda:3"
        calls.append("guard")
        if fails:
            raise RuntimeError("unsafe kernel")

    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu_worker", SimpleNamespace(Worker=Worker)
    )
    monkeypatch.setattr(guard, "configure_blackwell_gdn", configure)
    monkeypatch.delitem(sys.modules, "training.vllm_blackwell_worker", raising=False)
    try:
        module = importlib.import_module("training.vllm_blackwell_worker")
        worker = module.BlackwellGdnWorker()
        if fails:
            with pytest.raises(RuntimeError, match="unsafe kernel"):
                worker.init_device()
                worker.load_model()
            assert calls == ["device", "guard"]
        else:
            worker.init_device()
            worker.load_model()
            worker.profile_run()
            assert calls == ["device", "guard", "load", "profile"]
    finally:
        sys.modules.pop("training.vllm_blackwell_worker", None)


def test_output_kernel_drops_only_the_eight_warp_config(runtime, caplog):
    original = runtime.output.configs[:]
    with caplog.at_level("INFO"):
        assert guard.configure_blackwell_gdn("fla")
    assert runtime.output.configs == [c for c in original if c.num_warps != 8]
    assert all(any(c is old for old in original) for c in runtime.output.configs)
    assert {(c.kwargs["BK"], c.kwargs["BV"], c.num_warps) for c in runtime.output.configs} == {
        (64, 64, 4), (32, 32, 2)
    }
    assert runtime.output.cache_results is True
    assert "chunk_fwd_kernel_o without eight warps safe_configs=2" in caplog.text
    # Idempotent: a second worker setup keeps the restricted space and its cache.
    runtime.output.cache["shape"] = runtime.output.configs[0]
    assert guard.configure_blackwell_gdn("fla")
    assert runtime.output.cache == {"shape": runtime.output.configs[0]}


def test_serving_backend_leaves_the_output_kernel_alone(runtime):
    before = runtime.output.configs
    runtime.versions["vllm"] = "0.29.1.dev166+g3bda1ba09.precompiled"
    runtime.versions["triton"] = "3.7.1"
    assert guard.configure_blackwell_gdn("vllm")
    assert runtime.output.configs is before
    assert guard._FLA_OUTPUT_MODULE not in runtime.imported


def test_changed_output_space_is_fatal_before_either_kernel_changes(runtime):
    runtime.output.configs.append(triton.Config({"BK": 64, "BV": 128}, num_warps=4, num_stages=3))
    state_before, output_before = runtime.tuner.configs, runtime.output.configs[:]
    with pytest.raises(RuntimeError, match="Unexpected Blackwell GDN output config space"):
        guard.configure_blackwell_gdn("fla")
    assert runtime.tuner.configs is state_before
    assert runtime.output.configs == output_before


def test_output_kernel_late_installation_is_fatal(runtime):
    unsafe = runtime.output.configs[0]
    runtime.output.cache["observed"] = unsafe
    with pytest.raises(RuntimeError, match="before the first kernel invocation"):
        guard.configure_blackwell_gdn("fla")
    assert unsafe in runtime.output.configs
    # The state kernel was not narrowed either: validation precedes every change.
    assert any(config.num_warps == 4 for config in runtime.tuner.configs)


def test_output_guard_tampering_is_detected(runtime):
    guard.configure_blackwell_gdn("fla")
    runtime.output.cache["unsafe"] = triton.Config({"BK": 128, "BV": 128}, num_warps=8, num_stages=3)
    with pytest.raises(RuntimeError, match="output guard was modified"):
        guard.configure_blackwell_gdn("fla")


def test_missing_output_kernel_is_a_contract_change(runtime):
    delattr(runtime.modules[guard._FLA_OUTPUT_MODULE], guard._OUTPUT_KERNEL)
    with pytest.raises(RuntimeError, match="chunk_fwd_kernel_o"):
        guard.configure_blackwell_gdn("fla")
