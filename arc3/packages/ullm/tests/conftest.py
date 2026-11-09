"""Backend test matrix: columns (backends) x rows (tests).

Every real backend is built through its factory, so factory lifecycle is
exercised as a side effect. Cost tiers are gated twice:

- Selection: expensive columns carry pytest marks; the default addopts
  (root pyproject) exclude `gpu` and `api`, so plain runs touch only free
  columns. `pytest -m gpu` / `-m api` select one tier.
- Safety: gpu fixtures importorskip vllm and *fail* (not skip) when gpu
  was explicitly selected but no GPU is visible; api fixtures require the
  explicit ULLM_API_TESTS=1 opt-in — exported API keys alone never spend
  money.

Columns provide `make_backend(script)`: scripted columns (stub) consume
the per-test script; real-LLM columns ignore it and return the shared
session backend. Tests skip rows a column can't support via the
capability flags.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import ullm.backend
import ullm.backends.stub
import ullm.tools.python
from ullm.backend_matrix import BackendColumn

# The GPU columns' model: small, and both a thinking model and a
# vision-language one, so every GPU column runs real reasoning — produced and
# parsed — and real image input. A hybrid (gated-delta-network) model like the
# production engines, served with their recurrent-cache settings.
GPU_TEST_MODEL = "Qwen/Qwen3.5-9B"
GPU_TEST_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
# The template the 9B is served with: Qwen3.6's, which the production 27B
# ships. The 9B's own drops the reasoning of every assistant message before
# the last user message, so no call after a new user prompt could extend the
# served ids by tokens (``ullm.continuation``); Qwen3.6's keeps it under
# ``preserve_thinking``, as production serves it. The one other difference
# between the two: Qwen3.6 writes a tool call's non-string arguments as JSON.
SERVED_TEMPLATE_REPO = "Qwen/Qwen3.6-27B"
SERVED_TEMPLATE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
# The victim-offload engine keeps the small instruct model its pool and tier
# are sized against: it reads the engine's offload counters, not a backend.
VICTIM_TEST_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _require_gpu(request: pytest.FixtureRequest) -> None:
    """Layer-2 guard for gpu columns. Skip quietly when gpu was not asked
    for (e.g. `-m ""` on a CI box); fail loudly when it *was* asked for but
    no GPU is visible — an all-skipped green run must not look like GPU
    validation."""
    pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        if "gpu" in (request.config.option.markexpr or ""):
            pytest.fail("gpu tests selected but no GPU is visible; run under e.g. `srun --gres=gpu:1 make test-gpu`")
        pytest.skip("no GPU available")


def _require_api_opt_in(env_key: str) -> str:
    """Layer-2 guard for $ columns: explicit opt-in required; once opted
    in, a missing key is a loud failure, not a skip."""
    if os.environ.get("ULLM_API_TESTS") != "1":
        pytest.skip("api tests require explicit opt-in: ULLM_API_TESTS=1")
    key = os.environ.get(env_key)
    if not key:
        pytest.fail(f"ULLM_API_TESTS=1 but {env_key} is not set")
    return key


#: The card a GPU column's engine may take. Each column runs on a GPU of its
#: own (``make test-gpu-srun``); the 9B's bf16 weights are ~19 GB of a
#: small card, so two columns cannot share one card.
GPU_FRACTION = 0.85


@pytest.fixture(scope="session")
def gpu_test_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The pinned 9B snapshot with ``SERVED_TEMPLATE_REPO``'s template: its
    files linked, the template's two carriers written anew
    (``chat_template.jinja``, and the copy in ``tokenizer_config.json``), so
    the engine, its image processor, the server and ullm's own renders all
    read the same template.

    Downloaded in a child process: the hub client's connections would
    otherwise live in this one, and warnings are errors here — a socket it
    leaves to the collector fails whichever test runs when it is collected."""
    fetch = (
        "import json, sys\n"
        "from huggingface_hub import hf_hub_download, snapshot_download\n"
        "model, revision, template_repo, template_revision = sys.argv[1:]\n"
        "print(json.dumps([snapshot_download(model, revision=revision),\n"
        "    hf_hub_download(template_repo, 'chat_template.jinja', revision=template_revision)]))\n"
    )
    fetched = subprocess.run(
        [
            sys.executable,
            "-c",
            fetch,
            GPU_TEST_MODEL,
            GPU_TEST_MODEL_REVISION,
            SERVED_TEMPLATE_REPO,
            SERVED_TEMPLATE_REVISION,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot_path, template_path = json.loads(fetched.stdout.strip().splitlines()[-1])
    snapshot = Path(snapshot_path)
    template = Path(template_path).read_text()
    served = tmp_path_factory.mktemp("qwen3.5-9b-served")
    carriers = ("chat_template.jinja", "tokenizer_config.json")
    for source in snapshot.iterdir():
        if source.name not in carriers:
            (served / source.name).symlink_to(source.resolve())
    (served / "chat_template.jinja").write_text(template)
    tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = template
    (served / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return str(served)


def _small_qwen_engine_args(model_dir: str) -> Any:
    import vllm

    # enforce_eager skips CUDA-graph capture: slower tokens, much faster startup.
    # The hybrid's prefix caching as the production engines have it
    # (``ullm.backends.deployment``): recurrent states cached in "align" mode,
    # every state block kept, so a continuation hits where its previous turn
    # ended. The reasoning parser rides the engine args, where the server
    # column's server reads it.
    return vllm.AsyncEngineArgs(
        model=model_dir,
        max_model_len=8192,
        gpu_memory_utilization=GPU_FRACTION,
        enforce_eager=True,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        prefix_cache_retention_interval=None,
        reasoning_parser="qwen3",
    )


#: Past reasoning rendered on every assistant message (``SERVED_TEMPLATE_REPO``).
_TEMPLATE_KWARGS = {"preserve_thinking": True}


def _qwen_backend_kwargs() -> dict[str, Any]:
    # The template calls tools in the XML format qwen3_coder parses.
    return {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "chat_template_kwargs": dict(_TEMPLATE_KWARGS),
    }


# Session-scoped engines: startup dominates, so each GPU column stands its
# stack up once, on a GPU of its own. loop_scope="session": engine background
# tasks are bound to the loop they start on, so every matrix test runs on the
# session loop.


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _vllm_direct_backend(request: pytest.FixtureRequest, gpu_test_model_dir: str):
    _require_gpu(request)
    import ullm.backends.vllm_factory as vf
    import ullm.backends.vllmx

    factory = vf.EngineFactory(
        engine_args=_small_qwen_engine_args(gpu_test_model_dir),
        backend_cls=ullm.backends.vllmx.VLLM,
        backend_kwargs=_qwen_backend_kwargs(),
        mode="direct",
    )
    backend = await factory.start()
    yield backend
    await factory.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _vllm_proxy_backend(request: pytest.FixtureRequest, gpu_test_model_dir: str):
    _require_gpu(request)
    import ullm.backends.vllm_factory as vf
    import ullm.backends.vllmx

    factory = vf.EngineFactory(
        engine_args=_small_qwen_engine_args(gpu_test_model_dir),
        backend_cls=ullm.backends.vllmx.VLLM,
        backend_kwargs=_qwen_backend_kwargs(),
        mode="proxy",
    )
    backend = await factory.start()
    yield backend
    await factory.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _vllm_server_backend(request: pytest.FixtureRequest, gpu_test_model_dir: str):
    _require_gpu(request)
    import ullm.backends.vllm_factory as vf

    factory = vf.InprocessServerFactory(
        engine_args=_small_qwen_engine_args(gpu_test_model_dir),
        serve_overrides={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder"},
        backend_kwargs={"parse_think_tags": True, "chat_template_kwargs": dict(_TEMPLATE_KWARGS)},
    )
    backend = await factory.start()
    yield backend
    await factory.close()


# Pool pinned by block count rather than squeezed via gpu_memory_utilization,
# so the pressure point is a known number on any node. It must hold a good
# many requests at once: a pool that fits only one serialises the load however
# much of it is offered, and then nothing is ever evicted under contention.
VICTIM_POOL_BLOCKS = 512
# Well under the pool, so the engine's own reported max concurrency is
# comfortably above one and the scheduler has room to overlap requests.
VICTIM_MAX_MODEL_LEN = 1024
# Host tier several times the pool, so a miss on it means the connector never
# wrote rather than that it wrapped. This model's KV runs ~112 KB per token —
# a 16-token block is ~1.8 MB — and finish-holds store every context that ends
# under pressure, so the tier must hold the test's full churn, not a handful
# of head-spills.
VICTIM_CPU_TIER_GIB = 4.0


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def victim_offload_engine(request: pytest.FixtureRequest):
    """A raw vLLM engine with a small KV pool and the victim host tier.

    Raw rather than backend-wrapped: what these tests read is the engine's own
    offload counters, and no backend sits between them.
    """
    _require_gpu(request)
    import ullm.backends.config_local as config_local
    import vllm

    engine_args = vllm.AsyncEngineArgs(
        model=VICTIM_TEST_MODEL,
        max_model_len=VICTIM_MAX_MODEL_LEN,
        gpu_memory_utilization=0.30,
        num_gpu_blocks_override=VICTIM_POOL_BLOCKS,
        # Nothing can spill without it: a block is only offloadable once it is
        # full and hashed, and that is what prefix caching gives it.
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    config_local.enable_cpu_kv_offload(engine_args, cpu_gib=VICTIM_CPU_TIER_GIB, victim=True)
    engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
    yield engine
    engine.shutdown()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _openrouter_backend():
    key = _require_api_opt_in("OPENROUTER_API_KEY")
    import ullm.backends.oai

    backend = ullm.backends.oai.OpenAIServer(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model="qwen/qwen3-30b-a3b-instruct-2507",
        send_back_reasoning="none",
        # SiliconFlow drops a tool result sent as content parts, ullm's form, so
        # the model answers without it; OpenRouter routes there whenever this
        # model's other providers are down or rate-limited.
        request_extra_body={"provider": {"ignore": ["SiliconFlow"]}},
    )
    yield backend
    await backend.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _anthropic_backend():
    key = _require_api_opt_in("ANTHROPIC_API_KEY")
    import ullm.backends.anthropic

    # haiku-4-5 rejects adaptive thinking ("not supported on this model");
    # run the column thinking-free — reasoning passback still exercises the
    # input path, there is just no reasoning content to send.
    backend = ullm.backends.anthropic.Anthropic(api_key=key, model="claude-haiku-4-5", set_adaptive_thinking=False)
    yield backend
    print(
        f"\n[anthropic column] {backend.total_usage.to_string()}; cost ${backend.pricing.cost(backend.total_usage):.4f}"
    )
    await backend.close()


_COLUMN_PARAMS = [
    pytest.param("dummy", id="dummy"),
    pytest.param("stub", id="stub"),
    pytest.param("vllm-direct", id="vllm-direct", marks=pytest.mark.gpu),
    pytest.param("vllm-proxy", id="vllm-proxy", marks=pytest.mark.gpu),
    pytest.param("vllm-server", id="vllm-server", marks=pytest.mark.gpu),
    pytest.param("openrouter", id="openrouter", marks=pytest.mark.api),
    pytest.param("anthropic", id="anthropic", marks=pytest.mark.api),
]


@pytest.fixture(params=_COLUMN_PARAMS)
def backend_column(request: pytest.FixtureRequest) -> BackendColumn:
    name = request.param
    if name == "dummy":
        return BackendColumn(
            name=name,
            make_backend=lambda script: ullm.backend.DummyBackend(),
            real_llm=False,
            tool_calls=False,
            reasoning=False,
        )
    if name == "stub":
        return BackendColumn(
            name=name,
            make_backend=lambda script: ullm.backends.stub.StubBackend.from_script(*(script or [])),
            real_llm=False,
            tool_calls=True,
            reasoning=False,
            images=True,  # wiring check: ImageContent flows through unharmed
            response_injection=True,  # serves the prefix onto a scripted answer
        )
    # Real columns: the session backend is shared; per-test scripts don't apply.
    # Fixtures named through __name__, not string literals: pytest resolves
    # by name, so a rename that missed this table would fail at run time.
    fixture_name = {
        "vllm-direct": _vllm_direct_backend.__name__,
        "vllm-proxy": _vllm_proxy_backend.__name__,
        "vllm-server": _vllm_server_backend.__name__,
        "openrouter": _openrouter_backend.__name__,
        "anthropic": _anthropic_backend.__name__,
    }[name]
    backend = request.getfixturevalue(fixture_name)
    return BackendColumn(
        name=name,
        make_backend=lambda script: backend,
        real_llm=True,
        tool_calls=True,
        # openrouter column runs an instruct model with reasoning passback off
        reasoning=(name != "openrouter"),
        # a direct-engine backend refuses to pickle by design
        picklable=(name != "vllm-direct"),
        # vLLM V1 prefix-caches by default; Anthropic caches via the
        # backend's cache_control breakpoints. OpenRouter caching is
        # provider-dependent — not asserted.
        prefix_cache=(name != "openrouter"),
        # The GPU test model is a VL model; Anthropic is natively
        # multimodal; the openrouter column runs a text-only model.
        images=(name != "openrouter"),
        # Only the in-process engine appends caller text as prompt ids; the
        # proxy column reaches one.
        response_injection=name in ("vllm-direct", "vllm-proxy"),
        # Likewise the answer's sampling: two requests over the same ids.
        after_reasoning=name in ("vllm-direct", "vllm-proxy"),
    )


@pytest.fixture(scope="session")
def kernel_pool():
    return ullm.tools.python.KernelPool()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Backends with known pricing report their own cost at teardown (the
    # anthropic column above; LLMAgentSolver prints per-run usage+cost).
    # OpenRouter pricing varies per model and is not configured, hence the
    # dashboard pointer.
    if os.environ.get("ULLM_API_TESTS") == "1":
        print(
            "\n[api tests ran with ULLM_API_TESTS=1 — openrouter spend is on the dashboard; anthropic cost printed above]"
        )
