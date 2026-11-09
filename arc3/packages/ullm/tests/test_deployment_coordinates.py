"""Engine x environment as coordinates: what each contributes, and what a
combination nobody has served does.

``gpu_memory_utilization`` is measured on a pair rather than derived from
either half, so the last chain's job is to refuse a cell it has no measurement
for. A guess there is silent: the engine starts, profiles a pool sized for
another card, and the run finds out under load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("vllm", reason="needs the vllm extra (linux)")

import ullm.backends.deployment  # noqa: E402


def _engine_args(engine: str, environment: str) -> object:
    return ullm.backends.deployment.build(engine, environment).factory.engine_args  # type: ignore[union-attr]


def test_the_engine_decides_the_checkpoint_and_the_environment_the_gpus() -> None:
    """Each coordinate reaches the engine args, and neither reaches the other's."""
    slurm = _engine_args("qwen3.8-27b", "slurm-b200")
    kaggle = _engine_args("qwen3.8-27b", "kaggle-rtx")
    # One engine, so the checkpoint's own facts hold across both environments.
    assert slurm.kv_cache_dtype == kaggle.kv_cache_dtype  # type: ignore[attr-defined]
    assert slurm.reasoning_parser == kaggle.reasoning_parser  # type: ignore[attr-defined]
    # One environment each, so where the weights are and how much card they may
    # take are the environment's and the pair's.
    assert slurm.model == "Qwen/Qwen3.8-27B-FP8"  # type: ignore[attr-defined]
    assert kaggle.model.startswith("/kaggle/input/")  # type: ignore[attr-defined]
    assert slurm.gpu_memory_utilization != kaggle.gpu_memory_utilization  # type: ignore[attr-defined]


def test_two_engines_in_one_environment_differ_only_by_the_checkpoint() -> None:
    a = _engine_args("qwen3.8-27b", "slurm-b200")
    b = _engine_args("qwen3.6-27b", "slurm-b200")
    assert a.model != b.model  # type: ignore[attr-defined]
    assert a.tensor_parallel_size == b.tensor_parallel_size  # type: ignore[attr-defined]
    assert a.gpu_memory_utilization == b.gpu_memory_utilization  # type: ignore[attr-defined]


def test_a_mounted_checkpoint_drops_the_published_revision() -> None:
    """A revision pins the published repository; a mounted copy is immutable
    by being published, and pinning one the mount does not carry fails load."""
    assert _engine_args("qwen3.8-27b", "kaggle-rtx").revision is None  # type: ignore[attr-defined]
    assert _engine_args("qwen3.8-27b", "slurm-b200").revision is not None  # type: ignore[attr-defined]


@pytest.mark.parametrize("engine", ["qwen3.8-27b", "qwen3.6-27b", "qwen3.8-flash-next"])
def test_the_four_l4_cell_shards_the_rtx_cells_checkpoint_at_its_own_headroom(engine: str) -> None:
    """The pair block is where the second Kaggle box states its own memory
    figure rather than borrowing the RTX cell's: the same mounted snapshot,
    four shards, and a pool sized for a runner that reserves nothing for
    graph capture."""
    rtx = _engine_args(engine, "kaggle-rtx")
    l4 = _engine_args(engine, "kaggle-4xl4")
    assert l4.model == rtx.model  # type: ignore[attr-defined]
    assert l4.tensor_parallel_size * l4.pipeline_parallel_size == 4  # type: ignore[attr-defined]
    # The 27Bs keep a reserve outside the budget; the Flash cell states its
    # own measured figure, and nothing here exceeds the RTX.
    assert l4.gpu_memory_utilization <= rtx.gpu_memory_utilization  # type: ignore[attr-defined]
    assert l4.revision is None  # type: ignore[attr-defined]


def test_the_four_l4_flash_cell_takes_the_rtx_cells_serving_choices_on_its_own_layout() -> None:
    """The RTX Flash cell's serving choices (the mount, the MoE backend the
    older capability family also admits, the table on the host, the drafter),
    at the small card's own headroom -- an absolute margin, so under the
    RTX's fraction -- and over two stages of two ranks, which the dense 27B
    beside it does not need. It loads with the engine's own prefetch, where
    the RTX maps lazily behind the kernel's bounded reader: that reader
    moves with the loaders it has seen, and four of them do not walk the
    shards together."""
    l4 = _engine_args("qwen3.8-flash-next", "kaggle-4xl4")
    rtx = _engine_args("qwen3.8-flash-next", "kaggle-rtx")
    for name in ("model", "kernel_config", "engram_config", "speculative_config"):
        assert getattr(l4, name) == getattr(rtx, name), name
    assert (l4.safetensors_load_strategy, rtx.safetensors_load_strategy) == ("prefetch", "lazy")  # type: ignore[attr-defined]
    l4_deployment = ullm.backends.deployment.build("qwen3.8-flash-next", "kaggle-4xl4")
    assert l4_deployment.kaggle_engine is not None and l4_deployment.kaggle_engine.prefetch == ()
    l4_27b = _engine_args("qwen3.8-27b", "kaggle-4xl4")
    assert l4.gpu_memory_utilization < rtx.gpu_memory_utilization  # type: ignore[attr-defined]
    # Two stages of two ranks, not a four-way split, for the one engine whose
    # experts and KV heads split badly four ways; the dense 27B shards cleanly.
    assert (l4.tensor_parallel_size, l4.pipeline_parallel_size) == (2, 2)  # type: ignore[attr-defined]
    assert (l4_27b.tensor_parallel_size, l4_27b.pipeline_parallel_size) == (4, 1)  # type: ignore[attr-defined]


def test_the_kaggle_flash_cells_bound_the_indexers_prefill_buffer() -> None:
    """The sparse-attention indexer's prefill scoring buffer grows with the
    context and the start-up profile reserves none of it; on the Kaggle cards
    the cells chunk it, and no other cell does."""
    for environment in ("kaggle-4xl4", "kaggle-rtx"):
        kaggle = ullm.backends.deployment.build("qwen3.8-flash-next", environment).factory
        assert kaggle.env_overrides["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] == "64", environment  # type: ignore[union-attr]
    for engine, environment, serving in (
        ("qwen3.8-flash-next", "slurm-b200", "direct"),
        ("qwen3.8-27b", "kaggle-4xl4", "direct"),
        ("qwen3.8-27b", "kaggle-rtx", "direct"),
        ("qwen3.8-27b", "kaggle-4xl4", "server"),
        ("qwen3.8-27b", "kaggle-rtx", "server"),
    ):
        env = ullm.backends.deployment.build(engine, environment, serving=serving).factory.env_overrides  # type: ignore[union-attr,arg-type]
        assert "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" not in env, (engine, environment, serving)


@pytest.mark.parametrize(
    "engine, serving",
    [("qwen3.8-27b", "direct"), ("qwen3.8-27b", "server"), ("qwen3.8-flash-next", "direct")],
)
def test_the_four_l4_box_derives_no_dense_graph_ladder(engine: str, serving: str) -> None:
    """Dense capture is a rung per decode width, captured once per card under
    tensor parallelism; on cards whose headroom capture memory already set,
    the box does not derive one. A single card keeps the dense one."""
    l4 = ullm.backends.deployment.build(engine, "kaggle-4xl4", serving=serving).factory  # type: ignore[arg-type]
    rtx = ullm.backends.deployment.build(engine, "kaggle-rtx", serving=serving).factory  # type: ignore[arg-type]
    assert l4.dense_cuda_capture is False and rtx.dense_cuda_capture is True  # type: ignore[attr-defined]
    assert ullm.backends.deployment.build("qwen3.8-27b", "slurm-b200").factory.dense_cuda_capture is True  # type: ignore[attr-defined]


def test_the_four_l4_flash_cell_names_the_graphs_its_traffic_replays() -> None:
    """The drafted Flash-Next cell on the four L4s names its CUDA graph
    ladder: every request count from 12 to 20, and 28, exact for the verify
    rows (five tokens a request at four drafts) and the draft steps; nothing
    above 256 tokens. The 27B there keeps the engine's ladder."""
    sizes = _engine_args("qwen3.8-flash-next", "kaggle-4xl4").compilation_config.cudagraph_capture_sizes  # type: ignore[attr-defined]
    assert sizes[-1] == 256
    for requests in [*range(12, 21), 28]:
        assert requests in sizes and 5 * requests in sizes, requests
    assert _engine_args("qwen3.8-27b", "kaggle-4xl4").compilation_config.cudagraph_capture_sizes is None  # type: ignore[attr-defined]


def test_the_four_l4_box_widens_the_drivers_work_queues_for_every_model() -> None:
    """A pipeline stage's token receive spins on a side stream; at the
    driver's default eight hardware queues a CUDA graph's branch can wait
    behind it. The box sets the maximum, 32, whatever it serves; no other
    box sets it."""
    for engine, serving in (("qwen3.8-flash-next", "direct"), ("qwen3.8-27b", "direct"), ("qwen3.8-27b", "server")):
        env = ullm.backends.deployment.build(engine, "kaggle-4xl4", serving=serving).factory.env_overrides  # type: ignore[union-attr,arg-type]
        assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "32", (engine, serving)
    for environment in ("slurm-b200", "kaggle-rtx"):
        env = ullm.backends.deployment.build("qwen3.8-flash-next", environment).factory.env_overrides  # type: ignore[union-attr]
        assert "CUDA_DEVICE_MAX_CONNECTIONS" not in env, environment


def test_the_flash_cells_serve_the_throughput_stack_where_it_runs() -> None:
    """The stack's three parts are disjoint and make up the whole. The RTX
    cell serves all of it; the four-L4 cell its portable part alone, since the
    engine refuses the fused commit under pipeline stages and the SM120 images
    load on no card of its; the B200 none of it."""
    deployment = ullm.backends.deployment
    parts = (
        deployment.FLASHNEXT_THROUGHPUT_PORTABLE_ENV,
        deployment.FLASHNEXT_THROUGHPUT_FUSED_COMMIT_ENV,
        deployment.FLASHNEXT_THROUGHPUT_SM120_ENV,
    )
    assert sum(len(part) for part in parts) == len(deployment.FLASHNEXT_THROUGHPUT_ENV)

    def env(environment: str) -> dict[str, str]:
        return deployment.build("qwen3.8-flash-next", environment).factory.env_overrides  # type: ignore[union-attr]

    assert deployment.FLASHNEXT_THROUGHPUT_ENV.items() <= env("kaggle-rtx").items()
    l4 = env("kaggle-4xl4")
    assert deployment.FLASHNEXT_THROUGHPUT_PORTABLE_ENV.items() <= l4.items()
    out_of_reach = (
        deployment.FLASHNEXT_THROUGHPUT_FUSED_COMMIT_ENV.keys() | deployment.FLASHNEXT_THROUGHPUT_SM120_ENV.keys()
    )
    assert not out_of_reach & l4.keys()
    assert not deployment.FLASHNEXT_THROUGHPUT_ENV.keys() & env("slurm-b200").keys()


def test_the_four_l4_flash_cell_splits_its_stages_27_21() -> None:
    """The second stage holds the lm head, the sampling and the drafter, so
    the first takes three more layers (the pool's peak). Nowhere else."""
    env = ullm.backends.deployment.build("qwen3.8-flash-next", "kaggle-4xl4").factory.env_overrides  # type: ignore[union-attr]
    assert env["VLLM_PP_LAYER_PARTITION"] == "27,21"
    for engine, environment in (("qwen3.8-27b", "kaggle-4xl4"), ("qwen3.8-flash-next", "kaggle-rtx")):
        other = ullm.backends.deployment.build(engine, environment).factory.env_overrides  # type: ignore[union-attr]
        assert "VLLM_PP_LAYER_PARTITION" not in other, (engine, environment)


def test_an_engine_that_names_its_threads_is_not_served_over_http() -> None:
    """Flash-Next's engine expects its client to name each conversation's
    thread, which puts the host tier in park mode, and the HTTP client names
    none: the tier would park nothing. The build refuses it on every box;
    an engine that names no threads serves over HTTP as before."""
    for environment in ("slurm-b200", "kaggle-rtx", "kaggle-4xl4"):
        with pytest.raises(ValueError, match="cannot be served over the server topology"):
            ullm.backends.deployment.build("qwen3.8-flash-next", environment, serving="server")
    for engine in ("qwen3.8-27b", "qwen3.6-27b"):
        ullm.backends.deployment.build(engine, "slurm-b200", serving="server")


def test_a_kernel_bounds_its_compile_parallelism_and_the_cluster_does_not() -> None:
    """A kernel that compiles a kernel runs one multi-GiB compiler per job in
    a cgroup the model and the tier already fill, so its job count is the
    environment's; the cluster's comes from the box."""
    for environment in ("kaggle-rtx", "kaggle-4xl4"):
        env = ullm.backends.deployment.build("qwen3.8-27b", environment).factory.env_overrides  # type: ignore[union-attr]
        assert env["FLASHINFER_NO_DOWNLOAD"] == "1" and env["MAX_JOBS"] == "4", environment
    slurm = ullm.backends.deployment.build("qwen3.8-27b", "slurm-b200").factory.env_overrides  # type: ignore[union-attr]
    assert "MAX_JOBS" not in slurm and "FLASHINFER_NO_DOWNLOAD" not in slurm


@pytest.mark.parametrize(
    "engine, environment, missing",
    [("nope", "slurm-b200", "unknown engine"), ("qwen3.8-27b", "nope", "unknown environment")],
)
def test_an_unknown_coordinate_names_itself(engine: str, environment: str, missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        ullm.backends.deployment.build(engine, environment)


def test_the_deploy_assets_ride_with_the_serving_it_takes() -> None:
    """A kernel needs the wheelhouse and the dataset that hold the model; a
    SLURM job needs neither, and saying so is what stops one leaking."""
    kaggle = ullm.backends.deployment.build("qwen3.8-27b", "kaggle-rtx")
    slurm = ullm.backends.deployment.build("qwen3.8-27b", "slurm-b200")
    assert kaggle.kaggle_engine is not None and kaggle.kaggle_dataset_sources
    assert slurm.kaggle_engine is None and slurm.kaggle_dataset_sources == ()
    # The mount path IS the dataset ref, prefixed -- two views of one fact.
    assert kaggle.factory.engine_args.model.endswith(kaggle.kaggle_dataset_sources[0])  # type: ignore[union-attr]


def test_the_client_facts_are_the_engines_and_the_servers() -> None:
    """A caller reaching a served model builds their own backend; what they
    should not have to reproduce is which reasoning field a thinking model
    answers in, or that streaming loses the token ids continuation reads."""
    kwargs = ullm.backends.deployment.client_kwargs("qwen3.8-27b")
    assert kwargs["reasoning_history_field"] == "reasoning"
    assert kwargs["streaming"] is False
    assert kwargs["chat_template_kwargs"]["preserve_thinking"] is True


def test_a_named_checkpoint_is_served_and_the_cell_kept(tmp_path: Path) -> None:
    """The B200 Flash-Next cell serves a checkpoint the caller names in place
    of its own, every other fact of the cell the same."""
    own = _engine_args("qwen3.8-flash-next", "slurm-b200")
    named = ullm.backends.deployment.build("qwen3.8-flash-next", "slurm-b200", checkpoint=str(tmp_path))
    args = named.factory.engine_args  # type: ignore[union-attr]
    assert args.model == str(tmp_path)  # type: ignore[attr-defined]
    assert args.revision is None and args.tokenizer_revision is None  # type: ignore[attr-defined]
    args.model = own.model  # type: ignore[attr-defined]
    assert repr(args) == repr(own)


def test_a_named_checkpoint_needs_no_hub_cache_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline, vLLM resolves the model the argument object is built with to
    a cached snapshot: a cell built around the Hub repository fails without
    one, and one built around the named directory does not."""
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hub-cache"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    built = ullm.backends.deployment.build("qwen3.8-flash-next", "slurm-b200", checkpoint=str(checkpoint))
    assert built.factory.engine_args.model == str(checkpoint)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "engine, environment",
    [("qwen3.8-flash-next", "kaggle-rtx"), ("qwen3.8-27b", "slurm-b200")],
)
def test_a_named_checkpoint_is_refused_where_no_cell_serves_one(engine: str, environment: str) -> None:
    """A Kaggle cell's checkpoint is a Kaggle model, and the 27B cells were
    never asked for one: refused rather than written over a cell nobody ran."""
    with pytest.raises(ValueError, match="checkpoint"):
        ullm.backends.deployment.build(engine, environment, checkpoint="/data/shared/research/checkpoints/x")
