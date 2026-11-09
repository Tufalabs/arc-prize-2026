"""The host KV tier: four factory fields, resolved into a connector at start.

Covers what the deployment snapshot cannot reach — the connector, role, module
path and byte count those fields produce, which do not exist until a factory
starts — that both start paths actually build it and hand it to vLLM with the
run's job directory inside, and that the fields are what a deployment states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

vllm = pytest.importorskip("vllm", reason="needs the vllm extra (linux)")

import ullm.backends.config_local  # noqa: E402
import ullm.backends.vllm_factory  # noqa: E402
import ullm.backends.vllmx  # noqa: E402


def _factory(**fields: Any) -> "ullm.backends.vllm_factory.EngineFactory":
    """A factory over a throwaway engine spec, for the tier fields alone."""
    return ullm.backends.vllm_factory.EngineFactory(
        engine_args=vllm.AsyncEngineArgs(model="facebook/opt-125m"),
        backend_cls=ullm.backends.vllmx.VLLM,
        **fields,
    )


def test_the_tier_is_not_written_until_start() -> None:
    """Construction states the tier; start builds it. The connector's import —
    and the vLLM-line check inside it — belongs to the process that serves."""
    factory = _factory(cpu_kv_offload=True)
    assert factory.engine_args.kv_transfer_config is None
    factory._apply_cpu_kv_offload(factory.engine_args)
    assert factory.engine_args.kv_transfer_config is not None


def test_off_writes_nothing() -> None:
    factory = _factory()
    assert factory.cpu_kv_offload is False, "a factory asks for no tier unless told"
    factory._apply_cpu_kv_offload(factory.engine_args)
    assert factory.engine_args.kv_transfer_config is None


def test_the_resolved_connector_is_what_it_always_was() -> None:
    """The shipped tier, spelled out."""
    factory = _factory(cpu_kv_offload=True)
    factory._apply_cpu_kv_offload(factory.engine_args)
    config = factory.engine_args.kv_transfer_config
    assert config is not None
    assert config.kv_connector == "VictimOffloadingConnector"
    assert config.kv_role == "kv_both"
    assert config.kv_connector_module_path == "ullm.backends.victim_offload"
    extra = config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == int(ullm.backends.config_local.DEFAULT_CPU_KV_OFFLOAD_GIB * 1024**3)
    assert "eviction_policy" in extra, "the victim connector's host policy rides with it"


def test_the_size_and_the_connector_are_the_fields_that_choose_them() -> None:
    factory = _factory(cpu_kv_offload=True, cpu_kv_offload_gib=8.0, cpu_kv_offload_victim=False)
    factory._apply_cpu_kv_offload(factory.engine_args)
    config = factory.engine_args.kv_transfer_config
    assert config is not None
    assert config.kv_connector == "OffloadingConnector", "stock vLLM's, as victim=False asks"
    assert config.kv_connector_module_path is None, "in-tree: a set path would beat the registry"
    assert config.kv_connector_extra_config["cpu_bytes_to_use"] == int(8.0 * 1024**3)


def test_extra_settings_reach_the_connector() -> None:
    factory = _factory(cpu_kv_offload=True, cpu_kv_offload_extra={"engine_log": True})
    factory._apply_cpu_kv_offload(factory.engine_args)
    config = factory.engine_args.kv_transfer_config
    assert config is not None
    assert config.kv_connector_extra_config["engine_log"] is True
    assert "cpu_bytes_to_use" in config.kv_connector_extra_config, "merged, not replaced"


def test_another_connector_is_refused_rather_than_displaced() -> None:
    """Two connectors cannot both hold the KV, and the one silently dropped
    would be the one someone configured deliberately."""
    import vllm.config.kv_transfer

    factory = _factory(cpu_kv_offload=True)
    factory.engine_args.kv_transfer_config = vllm.config.kv_transfer.KVTransferConfig(
        kv_connector="SharedStorageConnector", kv_role="kv_both"
    )
    with pytest.raises(ValueError, match="kv_transfer_config is already set"):
        factory._apply_cpu_kv_offload(factory.engine_args)


class _Handoff(Exception):
    """Raised where a start path hands its engine args to vLLM, carrying them:
    the last point this process sees the configuration, and the first point
    where it is the whole configuration."""

    def __init__(self, engine_args: Any) -> None:
        super().__init__("engine args handed off")
        self.engine_args = engine_args


def _intercept(engine_args: Any, *args: Any, **kwargs: Any) -> Any:
    raise _Handoff(engine_args)


def _quiet_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a start path's toolkit provisioning off this process. PATH,
    CUDA_HOME, the JIT thread count and a FlashInfer workspace are the
    worker's business, and none of them is what these tests read."""
    monkeypatch.setattr(ullm.backends.config_local, "_ensure_path_has_venv_bin", lambda: None)
    monkeypatch.setattr(ullm.backends.config_local, "_ensure_cuda_home", lambda: None)
    monkeypatch.setattr(ullm.backends.vllm_factory, "_set_jit_parallelism", lambda: None)
    monkeypatch.setattr(ullm.backends.vllm_factory, "_isolate_flashinfer_workspace", lambda: str(tmp_path / "fi"))
    # The two handoffs: the in-process engine's constructor, and the server's
    # conversion of engine args into its own namespace.
    monkeypatch.setattr(vllm.AsyncLLMEngine, "from_engine_args", staticmethod(_intercept))
    monkeypatch.setattr(ullm.backends.vllm_factory, "_serve_args_from_engine_args", _intercept)


def _direct(job_dir: Path) -> Any:
    """The in-process engine, with the tier on and a run to trace into."""
    return ullm.backends.vllm_factory.EngineFactory(
        engine_args=vllm.AsyncEngineArgs(model="facebook/opt-125m"),
        backend_cls=ullm.backends.vllmx.VLLM,
        mode="direct",
        cpu_kv_offload=True,
        job_dir=job_dir,
    )


def _server(job_dir: Path) -> Any:
    """vLLM's HTTP server in this process, same tier, same run."""
    return ullm.backends.vllm_factory.InprocessServerFactory(
        engine_args=vllm.AsyncEngineArgs(model="facebook/opt-125m"),
        cpu_kv_offload=True,
        job_dir=job_dir,
    )


@pytest.mark.parametrize("build", [_direct, _server], ids=["direct", "server"])
async def test_both_start_paths_hand_vllm_the_tier_with_its_job_dir(
    build: Callable[[Path], Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Through the real ``_start`` up to the handoff, so what is asserted is
    what an engine would have been built from. The failure this catches is
    silent twice over: a tier declared on the factory and never built starts
    an engine sized for a host tier it does not have, and a job directory
    attached before the connector exists leaves the run's trace off."""
    _quiet_start(monkeypatch, tmp_path)
    with pytest.raises(_Handoff) as caught:
        await build(tmp_path)._start()
    engine_args = caught.value.engine_args
    config = engine_args.kv_transfer_config
    assert config is not None, "the tier was declared and never built"
    assert config.kv_connector == "VictimOffloadingConnector"
    assert config.kv_connector_extra_config["job_dir"] == str(tmp_path)


@pytest.mark.parametrize(
    "build, expect_on, expect_gib",
    [
        (lambda: _duck().build(engine="qwen3.8-27b"), True, None),
        (lambda: _no_tier(_duck().build(engine="qwen3.8-27b")), False, None),
        (lambda: _duck().build(engine="qwen3.8-27b", environment="kaggle-rtx"), True, 60.0),
    ],
)
def test_the_duck_columns_state_their_tier_on_the_factory(
    build: Any, expect_on: bool, expect_gib: float | None
) -> None:
    factory: Any = build().llm_config.backend_factory
    assert factory.cpu_kv_offload is expect_on
    assert factory.cpu_kv_offload_gib == expect_gib
    assert factory.engine_args.kv_transfer_config is None, "still unwritten before start"


def _no_tier(agent: Any) -> Any:
    """The same column with the tier turned off after construction."""
    agent.llm_config.backend_factory.cpu_kv_offload = False
    return agent


def _duck() -> Any:
    """Imported late: ullm's tests do not otherwise depend on the duck."""
    pytest.importorskip("duck.presets", reason="needs the duck package")
    from duck.agent import DuckAgent

    return DuckAgent
