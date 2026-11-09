"""The traffic benchmark's tools under ``packages/duck/tools``: the launcher's
arms set what they name and compose in the order given, and the archive
turns a saved log into a provenance record -- the reading, the checks
attributed to it, the code and the workload -- or a recorded failure when
the log carries no reading.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOLS = REPO / "packages" / "duck" / "tools"

pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")


def _load(name: str) -> Any:
    """A tool as a module: tools are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch = _load("traffic_launch")
archive = _load("traffic_archive")


# --------------------------------------------------------------------------
# the launcher's arms
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workload(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A one-lane workload file: ``build`` reads the workload it names."""
    import numpy as np
    import ullm.traffic

    def ids(*values: int) -> Any:
        return np.asarray(values, dtype=np.int32)

    def call(input_ids: Any, output_ids: Any, finished_at: float, *, opens: bool = False) -> ullm.traffic.RecordedCall:
        return ullm.traffic.RecordedCall(
            input_ids=input_ids,
            output_ids=output_ids,
            images=(),
            finished_at=finished_at,
            continued=not opens,
            opens=opens,
            after=None,
        )

    first_in, first_out = ids(1, 2, 3, 4), ids(5, 6)
    second_in = np.concatenate([first_in, first_out, ids(7, 8)])
    run = ullm.traffic.RecordedRun(
        "a", (call(first_in, first_out, 10.0, opens=True), call(second_in, ids(9, 10), 20.0))
    )
    cut = ullm.traffic.build_workload([run], {}, horizon_tokens=1, snapshot=15.0)
    return cut.save(tmp_path_factory.mktemp("traffic") / "workload.b64")


def _agent(workload: pathlib.Path) -> Any:
    from duck.traffic import TrafficDuck

    return TrafficDuck.build(engine="qwen3.8-flash-next", environment="kaggle-4xl4", workload=str(workload))


def _factory(agent: Any) -> Any:
    return agent.llm_config.backend_factory


def _expandable(f: Any) -> bool:
    return (
        f.env_overrides["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        and f.engine_args.enable_cumem_allocator is True
        and f.engine_args.disable_custom_all_reduce is True
    )


# What each arm must leave on the factory. An arm with no line here is an
# arm nobody has read, which the first test refuses.
EXPECTED: dict[str, Callable[[Any], bool]] = {
    # v7's agent and pacing are the agent's own fields; the factory carries
    # its host tier's engine log.
    "v7_like": lambda f: f.cpu_kv_offload_extra == {"engine_log": True},
    "expandable": lambda f: _expandable(f) and "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" not in f.env_overrides,
    "indexer_default": lambda f: "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" not in f.env_overrides,
    "custom_ar_off": lambda f: f.engine_args.disable_custom_all_reduce is True,
    "gdn_decode_triton": lambda f: f.env_overrides["VLLM_GDN_DECODE_KERNEL"] == "triton",
    "bf16_state": lambda f: f.engine_args.mamba_ssm_cache_dtype == "bfloat16",
    "bf16_state_two_blocks": lambda f: (
        f.engine_args.mamba_ssm_cache_dtype == "bfloat16" and f.engine_args.max_num_batched_tokens == 3300
    ),
    "unit64": lambda f: f.engine_args.prefix_match_unit == 64,
    "utilization_096": lambda f: f.engine_args.gpu_memory_utilization == 0.96,
    "all_l4": lambda f: (
        _expandable(f)
        and f.env_overrides["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] == "64"
        and f.env_overrides["VLLM_GDN_DECODE_KERNEL"] == "triton"
        and f.engine_args.mamba_ssm_cache_dtype == "bfloat16"
        and f.engine_args.max_num_batched_tokens == 1700
        and f.engine_args.prefix_match_unit == 32
    ),
    "eager": lambda f: f.engine_args.enforce_eager is True,
    "no_mtp": lambda f: f.engine_args.speculative_config is None,
    "mtp1": lambda f: _drafts(f, 1),
    "mtp2": lambda f: _drafts(f, 2),
    "mtp3": lambda f: _drafts(f, 3),
    "mtp4": lambda f: _drafts(f, 4),
    "gdn_recoverssm": lambda f: f.engine_args.speculative_config["gdn_recoverssm"] is True,
    "prefix_match_unit_16": lambda f: f.engine_args.prefix_match_unit == 16,
    "indexer_fp8": lambda f: f.engine_args.attention_config.indexer_kv_dtype == "fp8",
    "fp8_dense": lambda f: f.engine_args.additional_config["online_fp8_linear"] == list(launch.FP8_DENSE_TIER1),
    "draft_top_k_top_p": lambda f: f.engine_args.speculative_config["draft_top_k_top_p"] is True,
    "relaxed_accept": lambda f: (
        f.engine_args.speculative_config["accept_threshold_single"] == 0.3
        and f.engine_args.speculative_config["accept_threshold_acc"] == 0.3
    ),
    "autoround": lambda f: (
        f.engine_args.model == launch.ullm.backends.deployment.FLASHNEXT_E307_AUTOROUND_CHECKPOINT
        and f.engine_args.quantization == "inc"
    ),
    "qad": lambda f: (
        f.engine_args.model == launch.ullm.backends.deployment.FLASHNEXT_E307_QAD_CHECKPOINT
        and f.engine_args.quantization == "modelopt_mixed"
    ),
}

# The arms an arm needs before it: RecoverSSM rides a drafted arm, and the
# four L4s do not draft.
BEFORE: dict[str, tuple[str, ...]] = {
    "gdn_recoverssm": ("mtp3",),
    "draft_top_k_top_p": ("mtp3",),
    "relaxed_accept": ("mtp3",),
}


def _drafts(f: Any, k: int) -> bool:
    speculative = f.engine_args.speculative_config
    return (
        speculative["method"] == "mtp"
        and speculative["num_speculative_tokens"] == k
        and speculative["model"] == f.engine_args.model + "-drafter"
    )


def test_every_arm_sets_what_it_names(workload: pathlib.Path) -> None:
    assert set(EXPECTED) | {"baseline"} == set(launch.ARMS)
    for name, holds in EXPECTED.items():
        agent = _agent(workload)
        for arm in (*BEFORE.get(name, ()), name):
            launch.ARMS[arm](agent)
        assert holds(_factory(agent)), name


def test_the_baseline_arm_leaves_the_column_as_it_ships(workload: pathlib.Path) -> None:
    agent, control = _agent(workload), _agent(workload)
    launch.ARMS["baseline"](agent)
    assert repr(_factory(agent).engine_args) == repr(_factory(control).engine_args)
    assert _factory(agent).env_overrides == _factory(control).env_overrides


def test_arms_compose_in_the_order_given(workload: pathlib.Path) -> None:
    """The later arm wins where two touch one setting: ``--arm`` repeats."""
    agent = _agent(workload)
    for arm in ("unit64", "all_l4"):
        launch.ARMS[arm](agent)
    assert _factory(agent).engine_args.prefix_match_unit == 32
    agent = _agent(workload)
    for arm in ("all_l4", "unit64"):
        launch.ARMS[arm](agent)
    assert _factory(agent).engine_args.prefix_match_unit == 64


# --------------------------------------------------------------------------
# the archive
# --------------------------------------------------------------------------

# Lines as the run prints them, in the log's order: TAAF's start-up block,
# the engine's pool line, the replay's header, a warm-up minute (which the
# reading leaves out), an allocator failure, the settle mark, two scored
# minutes -- one with both queue labels -- the settled line, the window's
# summary, and the run winding down after it.
LOG = """git status:
  arc2-platypus                    e0e88a5     clean  flash-traffic             Docs sweep for the traffic branch
  arc-prize-2026-vllm                    6de023f0b4  pinned  6de023f0b4021af480d2f2fdc6f1d30118424d70  (non-editable)
benchmark.solver: LLMAgentSolver(label='', agent=TrafficDuck(run_config=RunConfig(max_concurrent=60)))
(EngineCore pid=679) INFO 09-19 21:03:54 [kv_cache_utils.py:2042] GPU KV cache size: 1,588,592 tokens, Maximum concurrency for 65,536 tokens per request: 24.24x
[traffic] replaying 60 of 100 lanes from /kaggle/input/datasets/o/ds/payload/workload_100lanes_20260919.b64
[traffic] 60 lanes, 1200 s, 5.0 s between a lane's calls; the rates are counted from 300 s in
[traffic] 61s: 60 in flight, 1 returned; gen 19 tok/s, uncached prompt 12259 tok/s over the last 61s
[engine] 21:06:45 running=38 waiting=22(capacity:2,deferred:20) kv_cache=89.8% | 60s: gen=211tok/s prompt=13250tok/s(new=12183) prefix_hit=8.5% offload=+126/-0MB/s ext_hit=0.0% preempt=0
[rank0]:[W919 21:07:01.0 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 67108864 bytes (free: 17891328, total: 23659151360).
[traffic] 301s: 52 in flight, 400 returned; gen 550 tok/s, uncached prompt 180 tok/s over the last 60s
[engine] 21:15:45 running=52 waiting=14(capacity:14) kv_cache=94.0% | 60s: gen=550tok/s prompt=2100tok/s(new=1700) prefix_hit=10.0% offload=+50/-40MB/s ext_hit=95.0% preempt=0
[engine] 21:16:45 running=53 waiting=13(capacity:3,deferred:10) kv_cache=94.5% | 60s: gen=560tok/s prompt=2000tok/s(new=1600) prefix_hit=12.0% offload=+50/-40MB/s ext_hit=95.0% preempt=0
[traffic] settled 899 s from 301 s in: gen 552.4 tok/s, uncached prompt 175.6 tok/s
[traffic] window 1200 s: generated 600000 tokens (500.0 tok/s), uncached prompt 300000 (250.0 tok/s); 900 calls returned
[engine] 21:31:00 running=0 waiting=0 kv_cache=1.0% | 60s: gen=0tok/s prompt=0tok/s(new=0) prefix_hit=0.0% ext_hit=0.0% preempt=0
"""


def _archived(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, text: str, *args: str) -> dict[str, Any]:
    """Archive ``text`` as a saved console log into a private archive, and
    return its provenance record."""
    monkeypatch.setattr(archive, "ARCHIVE", tmp_path / "archive")
    monkeypatch.setattr(archive, "WORKLOADS", tmp_path / "workloads")
    (tmp_path / "workloads").mkdir()
    (tmp_path / "workloads" / "workload_100lanes_20260919.b64").write_bytes(b"the same bytes or no comparison")
    log = tmp_path / "console.log"
    log.write_text(text)
    archive.main([str(log), *args])
    (record,) = [p for p in (tmp_path / "archive").iterdir() if p.is_dir()]
    return json.loads((record / "provenance.json").read_text())


def test_a_saved_log_yields_the_reading_and_the_checks_attributed_to_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _archived(tmp_path, monkeypatch, LOG, "--label", "bf16", "--kind", "kaggle-4xl4-interactive")
    reading = record["reading"]
    assert reading["settled"] == {
        "settled_s": 899,
        "settled_at_s": 301,
        "generated_tok_s": 552.4,
        "uncached_prompt_tok_s": 175.6,
    }
    assert (reading["lanes"], reading["workload_lanes"], reading["settle_s"]) == (60, 100, 300)
    assert (reading["kv_cache_tokens"], reading["kv_full_window_requests"]) == (1588592, 24.24)
    # Two engine lines sit between the settle mark and the window's summary;
    # the warm-up minute and the wind-down are not the reading's.
    assert reading["engine_lines"] == 2
    checks = reading["checks"]
    assert checks["running_mean"] == 52.5
    assert checks["capacity_waits"] == [3, 8, 14], "the capacity count, not the whole waiting count"
    # Both labels of ``waiting=13(capacity:3,deferred:10)`` are read.
    assert (checks["capacity_max"], checks["deferred_max"]) == (14, 10)
    assert checks["prefix_hit_pct"] == 11.0
    assert checks["combined_hit_pct"] == pytest.approx(95.55, abs=0.06)
    assert reading["allocator_failures"] == 1
    # The code from the log's own start-up block, the workload by content.
    assert (record["code"]["commit"], record["code"]["state"], record["code"]["vllm"]) == (
        "e0e88a5",
        "clean",
        "6de023f0b4",
    )
    assert record["workload"]["name"] == "workload_100lanes_20260919.b64"
    assert len(record["workload"]["sha256"]) == 16
    # The row the table takes, and the index that lists every run.
    row = (pathlib.Path(record["archive"]) / "row.md").read_text()
    assert row.startswith("| bf16 | kaggle-4xl4-interactive | 60 | **552.4 / 175.6** | 52.5 | 3 / 8 / 14 | 10 |")
    assert row.rstrip().endswith(f"`{pathlib.Path(record['archive']).name}` |")
    assert row.strip() in capsys.readouterr().out
    (indexed,) = (tmp_path / "archive" / "index.jsonl").read_text().splitlines()
    assert json.loads(indexed)["label"] == "bf16"


def test_a_log_with_no_settled_line_is_archived_as_a_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that fell over is provenance too, but it is no row."""
    head = LOG.split("[traffic] 301s:")[0]
    record = _archived(tmp_path, monkeypatch, head, "--label", "fell_over")
    assert record["reading"]["settled"] is None
    assert record["reading"]["engine_lines"] == 0
    assert record["code"]["commit"] == "e0e88a5"
    assert "NO READING" in capsys.readouterr().out


def test_a_kernels_jsonl_log_reads_as_the_text_it_wraps(tmp_path: pathlib.Path) -> None:
    """Kaggle serves a kernel's log as a JSON list, one object per line with
    the printed text under ``data``; the archive reads it as the log."""
    lines = [
        "[traffic] replaying 60 of 100 lanes from /kaggle/input/x/payload/w.b64\n",
        "[traffic] settled 899 s from 301 s in: gen 552.4 tok/s, uncached prompt 175.6 tok/s\n",
    ]
    wrapped = "\n".join(
        f"{'[' if i == 0 else ','}{json.dumps({'stream_name': 'stdout', 'time': float(i), 'data': line})}"
        for i, line in enumerate(lines)
    )
    path = tmp_path / "kernel.log"
    path.write_text(wrapped + "\n]")
    text = archive._text(path)
    assert text == "".join(lines)
    assert archive.SETTLED.search(text) is not None
