"""The seal's engine coordinate: Flash-Next is the default, the 27B columns stay selectable.

The references are indifferent (they never construct through a column), so
what these pin is the deployment surface: a bare build serves Flash-Next, the
3.8-27B engine serves its model with the seal's analyzer sampling and effort
live, and the 3.6 engine selects its own weights, parsers and effort scale.

The sampling is compared against ``analyzer_sampling_profiles`` (at the
standard cap) rather than against its values. What every column owes is the
SHARED profile — an engine or an environment that quietly rolled its own is
the regression worth catching, and the numbers themselves are single-sourced
in ``presets.py`` with their reasons in ``evidence.md``.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import ullm.backends.oai
import ullm.backends.stub
import ullm.factory

from seal.agent import SealAgent

presets = pytest.importorskip("seal.presets", reason="needs the vllm extra (linux)")
deployment = pytest.importorskip("ullm.backends.deployment", reason="needs the vllm extra (linux)")


def test_the_36_family_is_exactly_what_the_old_default_was() -> None:
    """vrfai weights, analyzer play profile, no effort opt-in, no effort key."""
    ret = SealAgent.build(engine="qwen3.6-27b")
    factory = cast("Any", ret.llm_config.backend_factory)

    assert factory.engine_args.model == "vrfai/Qwen3.6-27B-FP8"
    assert ret.llm_config.sampling_profiles["play"] == presets.analyzer_sampling_profiles()["play"]
    assert "native_reasoning_effort" not in factory.backend_kwargs
    assert ret.reasoning_effort == SealAgent.reasoning_effort, "3.6 has no xhigh; the class default stands"


def test_a_caller_who_brings_their_own_backend_keeps_every_engine() -> None:
    """A supplied backend is someone else's engine: nothing here may write to
    it, and the cells that size the host tier hold a field only our own
    factory has. The default engine reaches both paths, so a cell that
    forgot either one takes every default build down at construction."""

    def supplied() -> ullm.factory.BackendFactory:
        return ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.StubBackend.from_script("unused"))

    columns: list[dict[str, Any]] = [{}, {"engine": "qwen3.8-27b"}, {"environment": "kaggle-rtx"}]
    for kwargs in columns:
        agent = SealAgent.build(
            derive_max_model_len=False,
            derive_max_num_seqs=False,
            derive_max_pixels=False,
            backend_factory=supplied(),
            **kwargs,
        )  # type: ignore[arg-type]
        assert not hasattr(agent.llm_config.backend_factory, "cpu_kv_offload_gib"), kwargs
        assert agent.run_config.max_concurrent > 0, kwargs


def test_the_default_family_keeps_the_analyzer_sampling() -> None:
    """Official 3.8 weights, analyzer sampling and request cap, effort live."""
    ret = SealAgent.build(engine="qwen3.8-27b")
    factory = cast("Any", ret.llm_config.backend_factory)

    assert factory.engine_args.model == "Qwen/Qwen3.8-27B-FP8"
    assert ret.llm_config.sampling_profiles["play"] == presets.analyzer_sampling_profiles()["play"]
    assert factory.backend_kwargs["native_reasoning_effort"] is True
    assert ret.reasoning_effort == "xhigh", "the swept default rides the column"


def test_effort_is_preset_scoped_not_a_class_default() -> None:
    """The class default is provider-neutral — this field reaches whatever
    backend a hand-built seal is paired with, and "xhigh" is a level only
    the 3.8 family has. The presets inject xhigh where it was swept, and an
    explicit override still beats the injection."""
    assert (
        SealAgent(
            reply_cap_tokens=None,
            compaction_strategy="prune",
        ).reasoning_effort
        == "high"
    )
    assert SealAgent.build().reasoning_effort == "xhigh"
    assert SealAgent.build(engine="qwen3.6-27b").reasoning_effort == "high"
    assert SealAgent.build(reasoning_effort="low").reasoning_effort == "low"
    # The environment does not change which engine's effort scale applies.
    assert SealAgent.build(environment="kaggle-rtx").reasoning_effort == "xhigh"
    assert (
        SealAgent.build(engine="qwen3.6-27b", environment="kaggle-rtx").reasoning_effort == SealAgent.reasoning_effort
    )


def test_kaggle_builds_the_shipped_configuration_on_the_forked_engine() -> None:
    """Kaggle carries the fork, so no field of the agent is refused there.

    ``ullm.vllm_overlay`` replaces the stock engine the wheels install with
    the pinned revision, and refuses on the kernel rather than run a
    half-overlaid one; nothing is left for the constructor to guard. The
    shipped defaults are pinned here because a default that quietly turned
    the blank back on would change what a submission serves.
    """
    agent = SealAgent.build(engine="qwen3.8-27b", environment="kaggle-rtx")
    assert agent.zero_recurrent is False
    assert (
        agent.inject_text_on_first_prompt
        == SealAgent(
            reply_cap_tokens=None,
            compaction_strategy="prune",
        ).inject_text_on_first_prompt
    )
    assert SealAgent.build(engine="qwen3.8-27b", environment="kaggle-rtx", zero_recurrent=True).zero_recurrent is True


def test_the_kaggle_wheelhouse_is_a_stock_engine_of_this_line() -> None:
    """The offline closure and the engine this repo pins must be the same
    vLLM release, or the bundled source runs against a different engine than
    it was tested on. Stated as major.minor: the fork's pin carries a dev
    suffix and a local revision, and it is the release line that has to
    match, not the revision."""
    import ullm.kaggle
    import vllm

    wheel = ullm.kaggle.KAGGLE_VLLM_REQUIREMENTS[0]
    assert wheel.startswith("vllm==")
    installed = vllm.__version__.split("+")[0].split(".dev")[0]
    assert wheel.removeprefix("vllm==").split(".")[:2] == installed.split(".")[:2], (
        f"Kaggle installs {wheel} while this checkout runs vllm {vllm.__version__}"
    )


def test_the_engine_coordinate_selects_the_weights() -> None:
    """``SealAgent.build`` is the constructor callers use, and the engine key
    has to reach the weights through it -- a coordinate that resolves only
    inside ``presets`` is unreachable from the surface that matters."""

    def model_of(agent: SealAgent) -> str:
        """The engine's model id, through the factory the column built.

        A cast rather than an isinstance narrow: the factory class arrives
        via importorskip, which types as Any and narrows nothing.
        """
        factory = agent.llm_config.backend_factory
        assert factory is not None
        return cast("Any", factory).engine_args.model

    assert model_of(SealAgent.build()) == presets.JOINT205_CHECKPOINT
    assert model_of(SealAgent.build(engine="qwen3.8-27b")) == "Qwen/Qwen3.8-27B-FP8"
    assert model_of(SealAgent.build(engine="qwen3.6-27b")) == "vrfai/Qwen3.6-27B-FP8"


def test_joint205_drafts_with_the_combined_head_on_both_columns() -> None:
    """joint205's drafter is the combined ARC2/ARC3 head, not ullm's v1 head:
    from shared storage on the cluster, and on Kaggle its own model, mounted
    in place of the v1 head's."""
    import ullm.kaggle

    b200 = cast("Any", SealAgent.build().llm_config.backend_factory).engine_args
    assert b200.speculative_config["model"] == presets.ARC23_INT4_DRAFTER
    kaggle = SealAgent.build(environment="kaggle-rtx")
    rtx = cast("Any", kaggle.llm_config.backend_factory).engine_args
    assert rtx.speculative_config["model"] == ullm.kaggle.model_mount(presets.ARC23_INT4_DRAFTER_KAGGLE_MODEL)
    sources = kaggle.run_config.kaggle_model_sources
    assert presets.ARC23_INT4_DRAFTER_KAGGLE_MODEL in sources
    assert deployment.FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL not in sources


def test_zero_recurrent_runs_on_the_server_topology() -> None:
    """The fork declares the fields on its OpenAI request models and forwards
    them to SamplingParams, so a server carries the blank like the in-process
    engine does. The server must run the V2 model runner, which is why
    """
    agent = SealAgent.build(engine="qwen3.8-27b", serving="server", zero_recurrent=True)
    assert agent.zero_recurrent is True


def test_zero_recurrent_refuses_an_engine_without_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin that selects the fork is a workspace-ROOT setting, so a parent
    workspace consuming these packages resolves vllm by its own rules and can
    pair our agent with a stock build. Asking the installed engine is the only
    check that survives that, so it is the one that must hold."""
    import vllm

    class _StockSamplingParams:
        """A stand-in for a stock build: no zero_recurrent_state field.

        The real class is a msgspec Struct whose ``__struct_fields__`` is
        readonly, so the module attribute is swapped rather than the tuple.
        """

        __struct_fields__ = ("temperature", "top_p", "max_tokens")

    monkeypatch.setattr(vllm, "SamplingParams", _StockSamplingParams)
    with pytest.raises(ValueError, match="no SamplingParams.zero_recurrent_state"):
        SealAgent(reply_cap_tokens=None, compaction_strategy="prune", zero_recurrent=True).prepare_for_backend()
    # An agent that is not asking for the blank runs on a stock engine.
    agent = SealAgent(reply_cap_tokens=None, compaction_strategy="prune", zero_recurrent=False)
    agent.prepare_for_backend()
    assert agent.zero_recurrent is False


def test_external_server_must_run_this_clients_vllm_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server on another revision accepts the sampling parameters as unknown
    extras and serves the request unblanked, and the proxy the agent talks
    through hides what is behind it — so the mismatch has to be caught before
    the run starts, by asking the server what it is."""
    import json
    import urllib.request

    import vllm

    class _Response:
        """Minimal stand-in for what urlopen returns: json.load needs read()."""

        def __init__(self, version: str) -> None:
            self._payload = json.dumps({"version": version}).encode()

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _serving(version: str) -> Callable[..., _Response]:
        def _open(url: Any, timeout: Any = None) -> _Response:
            return _Response(version)

        return _open

    def _unreachable(url: Any, timeout: Any = None) -> _Response:
        raise OSError("connection refused")

    def _build(zero_recurrent: bool = True, **extra: Any) -> SealAgent:
        # The version check is what this tests, and only an agent asking
        # for the blank makes it, so a build here asks unless it says otherwise.
        return SealAgent.build(
            engine="qwen3.8-27b",
            environment="slurm-b200",
            derive_max_model_len=False,
            derive_max_num_seqs=False,
            derive_max_pixels=False,
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.oai.OpenAIServer(
                    base_url="http://host:8000/v1", model="m", **deployment.client_kwargs("qwen3.8-27b")
                )
            ),
            reply_cap_tokens=1024,
            zero_recurrent=zero_recurrent,
            **extra,
        )

    monkeypatch.setattr(urllib.request, "urlopen", _serving("0.0.0-not-our-fork"))
    with pytest.raises(ValueError, match="while this client is built against"):
        _build()

    # A server that cannot be asked is refused too: silence is not assent.
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable)
    with pytest.raises(ValueError, match="could not ask"):
        _build()

    # The matching build is accepted, and an agent not asking for the blank
    # never queries at all.
    monkeypatch.setattr(urllib.request, "urlopen", _serving(vllm.__version__))
    assert _build(zero_recurrent=True).zero_recurrent is True
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable)
    assert _build(zero_recurrent=False).zero_recurrent is False


def test_the_blank_needs_no_kv_connector() -> None:
    """The blank is asked for through the sampling parameter, not obtained
    from a connector's serving side effect: a factory built with the tier
    off asks for no host tier and is a valid seal. The tier is the column's
    default, so off is a field a caller turns off.
    """
    bare = SealAgent.build()
    cast("Any", bare.llm_config.backend_factory).cpu_kv_offload = False
    assert bare.zero_recurrent is SealAgent.zero_recurrent
    assert not cast("Any", bare.llm_config.backend_factory).cpu_kv_offload
    assert cast("Any", SealAgent.build().llm_config.backend_factory).cpu_kv_offload


def test_each_column_ships_the_host_tier_and_its_own_concurrency() -> None:
    """Both deployments run more games than their pool holds contexts, so each
    turns the victim tier on and runs a concurrency measured for its own cell.
    Sized differently: SLURM's B200 saturates deeper than the Kaggle card, and
    the Kaggle tier is capped by a cgroup the harness shares."""
    import ullm.backends.vllm_factory as vllm_factory
    from ullm.backends import config_local

    def tier_of(agent: SealAgent) -> typing.Any:
        agent_factory = agent.llm_config.backend_factory
        assert isinstance(agent_factory, vllm_factory.EngineFactory)
        return agent_factory

    slurm, kaggle = (
        SealAgent.build(engine="qwen3.8-27b"),
        SealAgent.build(engine="qwen3.8-27b", environment="kaggle-rtx"),
    )
    for agent in (slurm, kaggle):
        # Stated on the factory; the connector itself is built on the worker
        # at start (test_cpu_kv_offload_fields pins what that produces).
        assert tier_of(agent).cpu_kv_offload and tier_of(agent).cpu_kv_offload_victim is not False
        # The admission reserve that makes the engine queue rather than preempt.
        assert tier_of(agent).engine_args.watermark == deployment.ENGINE_ADMISSION_WATERMARK
    # None, not a number: config_local's shipped size is the one place it lives.
    assert tier_of(slurm).cpu_kv_offload_gib is None
    assert config_local.DEFAULT_CPU_KV_OFFLOAD_GIB > 0, "and it is a real size"
    assert tier_of(kaggle).cpu_kv_offload_gib is not None, "Kaggle sizes its own, against the cgroup"
    assert slurm.run_config.max_concurrent > kaggle.run_config.max_concurrent

    off = SealAgent.build(engine="qwen3.8-27b", environment="kaggle-rtx")
    tier_of(off).cpu_kv_offload = False
    assert not tier_of(off).cpu_kv_offload, "the tier is a field a caller turns off"


def test_a_named_checkpoint_takes_joint205s_place_beside_its_drafter(tmp_path: Path) -> None:
    """build(checkpoint=...) plays that directory where the B200 column plays
    joint205, drafted by the same combined head; Kaggle refuses one."""
    b200 = cast("Any", SealAgent.build(checkpoint=str(tmp_path)).llm_config.backend_factory).engine_args
    assert b200.model == str(tmp_path)
    assert b200.speculative_config["model"] == presets.ARC23_INT4_DRAFTER
    with pytest.raises(ValueError, match="checkpoint"):
        SealAgent.build(environment="kaggle-rtx", checkpoint=str(tmp_path))
    with pytest.raises(ValueError, match="backend_factory"):
        SealAgent.build(backend_factory=object(), checkpoint=str(tmp_path))
