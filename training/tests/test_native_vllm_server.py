"""Native serving contract tests using the real HTTP boundary and a fake engine."""

import asyncio
from copy import deepcopy
import importlib.metadata
import json
from types import SimpleNamespace as NS

from aiohttp.test_utils import TestClient, TestServer
import pytest

from engine_wire import generation
from training.engine_protocol import unpack_output

from training.lora_census import (
    CENSUS_RPC,
    WORKER_EXTENSION,
    adapter_modules,
    lora_census,
    read_adapter,
    served_modules,
)
from training.native_vllm_server import (
    ContractError,
    NativeService,
    create_app,
    validate_generation,
    tokenizer_fingerprint,
)


@pytest.fixture(autouse=True)
def fake_engine_version(monkeypatch):
    """Fake-engine HTTP metadata does not require installing CUDA's vLLM."""
    original = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.25.1" if name == "vllm" else original(name),
    )


def request(**changes):
    body = {
        "model": "base",
        "user": "run#1",
        "prompt": [1, 2],
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 4,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_p": 0.0,
        "min_tokens": 0,
        "logprobs": 0,
        "skip_special_tokens": False,
    }
    return body | changes


async def _generate(server, body):
    """Call the shared EngineLike seam with exact engine protocol fields."""
    from ullm.backends.engine_proxy import serialize_output

    body = dict(body)
    body.pop("arc3_lineage", None)
    wire = generation(body, server.adapter(body["model"]).identity())
    outputs = [
        item
        async for item in server.generate(
            wire["prompt"],
            wire["sampling_params"],
            wire["request_id"],
            lora_request=wire["lora_request"],
            request_context=wire["request_context"],
        )
    ]
    assert len(outputs) == 1
    return unpack_output(serialize_output(outputs[0]))


class Tokenizer:
    is_fast = True

    def convert_ids_to_tokens(self, ids, **kwargs):
        pieces = {3: "!!!!! malformed ", 4: "<tool"}
        return [pieces.get(token, str(token)) if token < 100 else "" for token in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    chat_template = "{{ messages }}"
    special_tokens_map = {"eos_token": "<end>"}
    bos_token_id, eos_token_id, pad_token_id = 0, 9, None
    max_token_id = 99
    backend_tokenizer = NS(
        to_str=lambda: json.dumps(
            {
                "model": {"vocab": {"a": 1, "b": 2}, "merges": ["a b"]},
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "decoder": {"type": "ByteLevel"},
                "added_tokens": [],
            }
        )
    )

    def __len__(self):
        return 100

    def encode(self, text, *, add_special_tokens):
        assert text == "0" and add_special_tokens is False
        return [5]

    def decode(self, ids, skip_special_tokens=False):
        # Token 2 is this vocabulary's tool-call marker.
        return "".join("<tool_call>" if token == 2 else str(token) for token in ids)


class LoraManager:
    """vLLM 0.25.1's LoRAModelManager as it wraps Qwen3_5ForConditionalGeneration,
    cut to two decoder layers: 0 Gated DeltaNet, 1 full attention, each with an
    MLP. The vision tower's linear layers make qkv, proj and linear_fc1 known
    LoRA suffixes, but without tower LoRA none of them is a LoRA layer."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "qkv": ["qkv"],
    }

    def __init__(self):
        from vllm.model_executor.models.utils import WeightsMapper

        self.model = NS(
            # Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
            hf_to_vllm_mapper=WeightsMapper(
                orig_to_new_prefix={
                    "model.visual.": "visual.",
                    "lm_head.": "language_model.lm_head.",
                    "model.language_model.": "language_model.model.",
                }
            ),
            lora_skip_prefixes=["mtp."],
        )
        layers = "language_model.model.layers"
        self.modules = dict.fromkeys(
            [f"{layers}.0.linear_attn.{name}" for name in ("in_proj_qkvz", "in_proj_ba", "out_proj")]
            + [f"{layers}.1.self_attn.{name}" for name in ("qkv_proj", "o_proj")]
            + [f"{layers}.{index}.mlp.{name}" for index in (0, 1) for name in ("gate_up_proj", "down_proj")]
        )
        # LoRAModelManager._register_packed_modules
        self.packed_modules = {}
        for layer in self.modules:
            prefix, _, name = layer.rpartition(".")
            children = self.packed_modules_mapping.get(name, [])
            if len(children) > 1:
                self.packed_modules[layer] = [f"{prefix}.{child}" for child in children]


GDN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
LAYER_MODULES = (
    [f"layers.0.linear_attn.{name}" for name in GDN]
    + [f"layers.1.self_attn.{name}" for name in ATTENTION]
    + [f"layers.{index}.mlp.{name}" for index in (0, 1) for name in MLP]
)


def write_adapter(directory, modules=LAYER_MODULES, *, rank=2, **config):
    """A runtime LoRA export (training/runtime_lora.py) of LoraManager's model.

    ``modules`` are under the decoder (``layers.0.mlp.up_proj``) unless they
    name a full tensor prefix (``base_model.model.``).
    """
    import torch
    from safetensors.torch import save_file

    tensors = {}
    for module in modules:
        name = (
            module
            if module.startswith("base_model.")
            else f"base_model.model.model.language_model.{module}"
        )
        tensors[f"{name}.lora_A.weight"] = torch.zeros(rank, 4)
        tensors[f"{name}.lora_B.weight"] = torch.zeros(4, rank)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": 2 * rank,
                "target_modules": sorted({module.rsplit(".", 1)[-1] for module in modules}),
                "rank_pattern": {},
                "alpha_pattern": {},
            }
            | config
        )
    )
    return str(directory)


# The census of the default adapter: packed modules under their packed layer.
APPLIED = {
    "down_proj": "down_proj",
    "gate_proj": "gate_up_proj",
    "in_proj_a": "in_proj_ba",
    "in_proj_b": "in_proj_ba",
    "in_proj_qkv": "in_proj_qkvz",
    "in_proj_z": "in_proj_qkvz",
    "k_proj": "qkv_proj",
    "o_proj": "o_proj",
    "out_proj": "out_proj",
    "q_proj": "qkv_proj",
    "up_proj": "gate_up_proj",
    "v_proj": "qkv_proj",
}


class Engine:
    def __init__(self):
        self.vllm_config = NS(
            model_config=NS(
                model="base/repository",
                revision="pinned-model",
                architectures=["Qwen3VLForConditionalGeneration"],
                tokenizer="base/tokenizer",
                tokenizer_revision="pinned-tokenizer",
                served_model_name="base",
                max_model_len=32,
                get_vocab_size=lambda: 100,
                generation_config="vllm",
                override_generation_config={},
                logprobs_mode="processed_logprobs",
                max_logprobs=20,
                logits_processors=[],
                get_diff_sampling_param=lambda: {},
                secret="never expose",
            ),
            cache_config=NS(kv_offloading_size=None),
            kv_transfer_config=None,
            # A LoRA engine; PolicyEngine below serves merged weights without LoRA.
            lora_config=NS(max_lora_rank=16, max_loras=2),
            parallel_config=NS(worker_extension_cls=WORKER_EXTENSION),
            offload_config=NS(
                uva=NS(cpu_offload_gb=0), prefetch=NS(offload_group_size=0)
            ),
            speculative_config=None,
        )
        self.calls = []
        self.added = []
        self.removed = []
        self.controls = []
        self.reset_ok = True
        self.aborted = []
        self.output_edit = lambda output: output
        self.entered = asyncio.Event()
        self.release = None
        self.error = None
        self.add_ok = self.remove_ok = True
        self.census_calls = []
        self.census_result = None

    def get_tokenizer(self):
        return Tokenizer()

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        """The census worker extension of one worker over LoraManager's model."""
        assert method == CENSUS_RPC and not kwargs
        self.census_calls.append(list(args[0]))
        if self.census_result is not None:
            return self.census_result
        return [served_modules(LoraManager(), *args)]

    async def generate(self, prompt, sampling, request_id, *, lora_request):
        self.calls.append((deepcopy(prompt), sampling, request_id, lora_request))
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.error:
            raise self.error
        tokens = [3] if sampling.max_tokens == 1 else [3, 4]
        output = NS(
            request_id=request_id,
            prompt_token_ids=list(prompt["prompt_token_ids"]),
            finished=True,
            lora_request=lora_request,
            num_cached_tokens=1,
            outputs=[
                NS(
                    index=0,
                    token_ids=tokens,
                    text="!!!!! malformed "
                    if sampling.max_tokens == 1
                    else "!!!!! malformed <tool",
                    finish_reason="length",
                    stop_reason=None,
                    logprobs=[
                        {
                            token: NS(rank=None, logprob=-0.4),
                            99: NS(rank=None, logprob=-0.1),
                        }
                        for token in tokens
                    ],
                )
            ],
        )
        yield self.output_edit(output)

    async def add_lora(self, adapter):
        self.added.append(adapter)
        self.controls.append(("add", adapter.lora_int_id))
        return self.add_ok

    async def remove_lora(self, identifier):
        self.removed.append(identifier)
        self.controls.append(("remove", identifier))
        return self.remove_ok

    async def reset_prefix_cache(self):
        self.controls.append(("reset", None))
        return self.reset_ok

    async def abort(self, request_id):
        self.aborted.append(request_id)

    async def check_health(self):
        if self.error:
            raise self.error


@pytest.mark.parametrize("image_limit", [0, 999])
def test_image_processor_is_loaded_before_service_readiness(monkeypatch, image_limit):
    """Cold image setup belongs to startup, not a collector's metadata deadline."""
    engine = Engine()
    engine.vllm_config.model_config.multimodal_config = NS(
        get_limit_per_prompt=lambda _modality: image_limit
    )
    monkeypatch.setattr(
        "training.native_vllm_server.tokenizer_fingerprint", lambda tokenizer: {}
    )
    loaded = []
    monkeypatch.setattr(
        NativeService, "_ensure_media_processor", lambda self: loaded.append(self)
    )
    ready = NativeService(
        engine, sampling_factory=lambda **kwargs: NS(**kwargs), lora_factory=NS
    )
    assert loaded == ([ready] if image_limit else [])


def test_image_pixel_ceiling_is_the_engines_or_the_processors():
    """What smart_resize shrinks above: the engine's max_pixels, else the
    processor's size["longest_edge"] (16,777,216 in Flash's preprocessor config)."""
    from training.native_vllm_server import image_pixel_ceiling

    processor = NS(image_processor=NS(size={"longest_edge": 16_777_216, "shortest_edge": 65_536}))
    assert image_pixel_ceiling(processor, {}) == 16_777_216
    assert image_pixel_ceiling(processor, {"max_pixels": 1_000_000}) == 1_000_000
    for unstated in (NS(image_processor=NS(size={"shortest_edge": 65_536})),
                     NS(image_processor=NS()), NS()):
        with pytest.raises(ContractError, match="pixel ceiling"):
            image_pixel_ceiling(unstated, {})
    for value in (0, -1, True, 1.5e6):
        with pytest.raises(ContractError, match="pixel ceiling"):
            image_pixel_ceiling(processor, {"max_pixels": value})


def _image_engine(image_limit, size):
    engine = Engine()
    engine.vllm_config.model_config.multimodal_config = NS(
        get_limit_per_prompt=lambda _modality: image_limit, mm_processor_kwargs={}
    )
    image_processor = NS(size=size, merge_size=2, to_dict=lambda: {"merge_size": 2})
    processor = NS(image_processor=image_processor, image_token_id=99)
    engine.renderer = NS(mm_processor=NS(info=NS(get_hf_processor=lambda **kwargs: processor)))
    return engine


@pytest.mark.parametrize("image_limit", [0, 999])
def test_metadata_states_the_served_image_pixel_ceiling(monkeypatch, image_limit):
    """Workers compare their largest render with it (None: images disabled)."""
    pytest.importorskip("ullm.media", reason="requires the private harness media identity")
    monkeypatch.setattr(
        "training.native_vllm_server.tokenizer_fingerprint", lambda tokenizer: {}
    )
    engine = _image_engine(image_limit, {"longest_edge": 16_777_216, "shortest_edge": 65_536})
    native = NativeService(engine, sampling_factory=lambda **kwargs: NS(**kwargs), lora_factory=NS)
    metadata = native.metadata()
    assert metadata["image_max_pixels"] == (16_777_216 if image_limit else None)
    # The ceiling stays outside ullm's processing identity, which the trainer
    # recomputes and compares exactly.
    assert "image_max_pixels" not in metadata.get("processing", {})


def test_image_server_without_a_pixel_ceiling_does_not_start(monkeypatch):
    pytest.importorskip("ullm.media", reason="requires the private harness media identity")
    monkeypatch.setattr(
        "training.native_vllm_server.tokenizer_fingerprint", lambda tokenizer: {}
    )
    with pytest.raises(ContractError, match="pixel ceiling"):
        NativeService(_image_engine(999, {}), sampling_factory=lambda **kwargs: NS(**kwargs),
                      lora_factory=NS)


def _summary_service(token_ids, monkeypatch):
    monkeypatch.setattr(
        "training.native_vllm_server.tokenizer_fingerprint", lambda tokenizer: {}
    )
    return NativeService(
        Engine(), sampling_factory=lambda **kwargs: NS(**kwargs), lora_factory=NS,
        summary_forbidden_token_ids=token_ids,
    )


def test_summary_forbidden_ids_must_decode_to_the_tool_call_marker(monkeypatch):
    """The ban is a literal copied from the Duck; startup checks it under the served tokenizer."""
    assert _summary_service((2,), monkeypatch).summary_forbidden_token_ids == {2}
    assert _summary_service((), monkeypatch).summary_forbidden_token_ids == frozenset()
    with pytest.raises(RuntimeError, match="decodes to '5'.*not '<tool_call>'"):
        _summary_service((2, 5), monkeypatch)
    for outside in (100, -1, True):
        with pytest.raises(RuntimeError, match="outside the model vocabulary"):
            _summary_service((outside,), monkeypatch)


def test_summary_forbidden_ids_option():
    from training.native_vllm_server import summary_forbidden_token_ids

    assert summary_forbidden_token_ids("248058") == (248058,)
    assert summary_forbidden_token_ids("248058,248059") == (248058, 248059)
    assert summary_forbidden_token_ids("") == summary_forbidden_token_ids(" ") == ()
    for invalid in ("-1", "1,1", "x", "1,"):
        with pytest.raises(ValueError):
            summary_forbidden_token_ids(invalid)


def service(engine=None, *, require_decoder=True, summary_forbidden_token_ids=(), policy_weights=False):
    """Build a fake service only when its shared runtime dependencies exist."""
    pytest.importorskip(
        "ullm.backends.engine_proxy", reason="requires the private harness transport"
    )
    if require_decoder:
        pytest.importorskip(
            "vllm.v1.engine.detokenizer",
            reason="requires the shared engine text decoder",
        )
    engine = Engine() if engine is None else engine
    result = NativeService(
        engine,
        sampling_factory=lambda **kwargs: NS(
            stop=None,
            include_stop_str_in_output=False,
            spaces_between_special_tokens=True,
            **kwargs,
        ),
        lora_factory=lambda name, identifier, path, **kwargs: NS(
            lora_name=name, lora_int_id=identifier, lora_path=path, **kwargs
        ),
        summary_forbidden_token_ids=summary_forbidden_token_ids,
        policy_weights=policy_weights,
    )
    return result, engine


def test_exact_integer_prompt_sampled_logprobs_and_malformed_text_survive():
    async def scenario():
        server, engine = service()
        result = await _generate(server, request())
        assert result == {
            "request_id": "run#1",
            "model": "base",
            "adapter": server.adapter("base").identity(),
            "prompt_token_ids": [1, 2],
            "output_token_ids": [3, 4],
            "output_logprobs": [-0.4, -0.4],
            "text": "!!!!! malformed <tool",
            "finish_reason": "length",
            "stop_reason": None,
            "num_cached_tokens": 1,
        }
        prompt, sampling, stamp, adapter = engine.calls[0]
        assert prompt == {"prompt_token_ids": [1, 2]} and adapter is None
        assert sampling.logprobs == 0 and sampling.skip_special_tokens is False
        assert sampling.presence_penalty == 0 and sampling.repetition_penalty == 1
        assert len(engine.calls) == 1 and server.failure is None

    asyncio.run(scenario())


REJECTED_REQUESTS = [
    {"presence_penalty": 1.5},
    {"frequency_penalty": 1},
    {"repetition_penalty": 1.1},
    {"min_p": 0.1},
    {"min_tokens": 1},
    {"min_tokens": False},
    {"logprobs": 1},
    {"logprobs": False},
    {"skip_special_tokens": True},
    {"ignore_eos": True},
    {"temperature": float("nan")},
    {"temperature": True},
    {"temperature": -1},
    {"top_p": 0},
    {"top_p": 1.1},
    {"top_k": -2},
    {"top_k": False},
    {"seed": False},
    {"max_tokens": True},
    {"max_tokens": 31},
    {"max_tokens": 0},
    {"prompt": [True]},
    {"prompt": [-1]},
    {"prompt": [100]},
    {"prompt": []},
    {"prompt": "text"},
    {"user": None},
    {"user": "unstamped"},
    {"user": "run#"},
    {"stream": False},
    {"return_token_ids": True},
    {"best_of": 1},
    {"echo": False},
    {"use_beam_search": False},
    {"stop": ["end"]},
    {"logit_bias": {"1": "-100"}},
    {"n": 2},
    {"n": True},
    {"messages": []},
    {"model": "unknown"},
]


@pytest.mark.parametrize("changes", REJECTED_REQUESTS)
def test_rejects_non_native_or_non_neutral_request_before_engine(changes):
    async def scenario():
        server, engine = service()
        with pytest.raises(ContractError):
            await _generate(server, request(**changes))
        assert engine.calls == [] and server.failure is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [change for change in REJECTED_REQUESTS if change.get("model") != "unknown"],
)
def test_pure_validator_rejects_non_native_or_non_neutral_request(changes):
    with pytest.raises(ContractError):
        validate_generation(request(**changes), max_model_len=32, vocab_size=100)


def test_pure_validator_preserves_neutral_sampling_and_normalizes_disabled_top_k():
    fields = validate_generation(request(top_k=0), max_model_len=32, vocab_size=100)
    assert fields["top_k"] == -1 and fields["logprobs"] == 0
    assert fields["max_tokens"] == 4 and fields["temperature"] == fields["top_p"] == 1.0
    assert fields["presence_penalty"] == 0 and fields["repetition_penalty"] == 1


def test_real_sampling_params_request_sampled_processed_logprobs_only():
    pytest.importorskip("vllm")
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    fields = validate_generation(request(top_k=0), max_model_len=32, vocab_size=100)
    params = SamplingParams(**fields, output_kind=RequestOutputKind.FINAL_ONLY)
    assert params.logprobs == 0 and params.output_kind == RequestOutputKind.FINAL_ONLY
    assert (
        params.top_k == -1
        and params.presence_penalty == 0
        and params.repetition_penalty == 1
    )


def _emit_token(engine, token):
    """Emit an exact model token with aligned evidence, even without text."""

    def replace_output(output):
        output.outputs[0].token_ids = [token]
        output.outputs[0].text = ""
        output.outputs[0].logprobs = [{token: NS(rank=None, logprob=-0.4)}]
        return output

    engine.output_edit = replace_output


def test_model_only_sampled_token_survives_capture_and_next_exact_prompt():
    async def scenario():
        engine = Engine()
        engine.vllm_config.model_config.get_vocab_size = lambda: 128
        server, engine = service(engine)
        _emit_token(engine, 127)
        first = await _generate(server, request())
        assert first["output_token_ids"] == [127]
        assert first["output_logprobs"] == [-0.4]
        prompt = first["prompt_token_ids"] + first["output_token_ids"] + [5]
        second = await _generate(server, request(user="run#2", prompt=prompt))
        assert second["prompt_token_ids"] == prompt
        assert engine.calls[1][0]["prompt_token_ids"] == [1, 2, 127, 5]
        assert server.failure is None
        with pytest.raises(ContractError, match="vocabulary"):
            await _generate(server, request(user="run#3", prompt=[128]))
        assert len(engine.calls) == 2
        _emit_token(engine, 128)
        with pytest.raises(RuntimeError, match="sampled token logprob"):
            await _generate(server, request(user="run#4"))
        assert server.failure is not None

    asyncio.run(scenario())


def test_sparse_tokenizer_only_ids_are_inputs_but_not_model_outputs():
    async def scenario():
        engine = Engine()
        tokenizer = Tokenizer()
        tokenizer.max_token_id = 255
        engine.get_tokenizer = lambda: tokenizer
        server, engine = service(engine)
        assert len(tokenizer) == 100
        first = await _generate(server, request(prompt=[255]))
        assert first["prompt_token_ids"] == [255]
        with pytest.raises(ContractError, match="vocabulary"):
            await _generate(server, request(user="run#2", prompt=[256]))
        assert len(engine.calls) == 1 and server.failure is None
        _emit_token(engine, 255)
        with pytest.raises(RuntimeError, match="sampled token logprob"):
            await _generate(server, request(user="run#3"))
        assert server.failure is not None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault",
    [
        "prompt",
        "request_id",
        "adapter",
        "logprob",
        "missing_logprob",
        "length",
        "budget",
        "finish",
        "cached",
        "text",
        "stop_reason",
        "boolean_stop_reason",
    ],
)
def test_engine_evidence_failure_poisons_all_future_draws(fault):
    async def scenario():
        server, engine = service()

        def corrupt(output):
            if fault == "prompt":
                output.prompt_token_ids = [2, 1]
            if fault == "request_id":
                output.request_id = "other#draw"
            if fault == "adapter":
                output.lora_request = NS(lora_int_id=1)
            if fault == "logprob":
                output.outputs[0].logprobs[0][3].logprob = float("nan")
            if fault == "missing_logprob":
                output.outputs[0].logprobs[0] = {}
            if fault == "length":
                output.outputs[0].logprobs = []
            if fault == "budget":
                output.outputs[0].token_ids *= 3
            if fault == "finish":
                output.outputs[0].finish_reason = "error"
            if fault == "text":
                output.outputs[0].text = "different from the sampled token IDs"
            if fault == "stop_reason":
                output.outputs[0].stop_reason = 8
            if fault == "boolean_stop_reason":
                output.outputs[0].stop_reason = True
            if fault == "cached":
                output.num_cached_tokens = 100
            return output

        engine.output_edit = corrupt
        with pytest.raises(RuntimeError):
            await _generate(server, request())
        with pytest.raises(RuntimeError, match="poisoned"):
            await _generate(server, request(user="run#2"))
        assert len(engine.calls) == 1

    asyncio.run(scenario())


def test_adapter_swap_keeps_actual_identity_and_retires_old_id(tmp_path):
    async def scenario():
        server, engine = service()
        adapter = write_adapter(tmp_path / "adapter")
        first = (await server.load_adapter({"name": "policy", "path": adapter}))["adapter"]
        assert first["id"] == 1
        generated = await _generate(server, request(model="policy"))
        assert generated["adapter"] == first
        assert engine.calls[0][3] is engine.added[0]
        second = (await server.load_adapter({"name": "policy", "path": adapter}))["adapter"]
        assert second["id"] == 2 and second != first and engine.removed == [1]
        assert server.metadata("policy")["adapter"] == second
        assert await server.unload_adapter({"name": "policy"}) == {"removed": True}
        with pytest.raises(ContractError, match="unknown model"):
            await _generate(server, request(model="policy", user="run#2"))
        assert len(engine.calls) == 1 and engine.removed == [1, 2]
        assert engine.controls == [
            ("add", 1),
            ("reset", None),
            ("add", 2),
            ("remove", 1),
            ("reset", None),
            ("remove", 2),
            ("reset", None),
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_failed_cache_reset_never_publishes_adapter_or_allows_another_draw(
    tmp_path, operation
):
    async def scenario():
        server, engine = service()
        adapter = write_adapter(tmp_path / "adapter")
        await server.load_adapter({"name": "policy", "path": adapter})
        original = server.adapter("policy")
        engine.reset_ok = False
        with pytest.raises(RuntimeError, match="prefix cache reset"):
            if operation == "load":
                await server.load_adapter({"name": "policy", "path": adapter})
            else:
                await server.unload_adapter({"name": "policy"})
        assert server.adapters["policy"] == original
        with pytest.raises(RuntimeError, match="poisoned"):
            await _generate(server, request(model="policy"))
        assert not engine.calls and not server.mutating

    asyncio.run(scenario())


def test_adapter_controls_and_duplicate_request_rejected_during_draw(tmp_path):
    async def scenario():
        server, engine = service()
        adapter = write_adapter(tmp_path / "adapter")
        await server.load_adapter({"name": "policy", "path": adapter})
        engine.release = asyncio.Event()
        task = asyncio.create_task(_generate(server, request(model="policy")))
        await engine.entered.wait()
        with pytest.raises(ContractError, match="duplicate"):
            await _generate(server, request(model="policy"))
        with pytest.raises(ContractError, match="drained"):
            await server.load_adapter({"name": "policy", "path": adapter})
        with pytest.raises(ContractError, match="drained"):
            await server.unload_adapter({"name": "policy"})
        assert len(engine.added) == 1 and engine.removed == []
        engine.release.set()
        await task
        assert len(engine.calls) == 1 and not server.inflight and server.failure is None

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["load", "retire"])
def test_uncertain_adapter_control_failure_poisons_engine(tmp_path, operation):
    async def scenario():
        server, engine = service()
        adapter = write_adapter(tmp_path / "adapter")
        if operation == "retire":
            await server.load_adapter({"name": "policy", "path": adapter})
            engine.remove_ok = False
        else:
            engine.add_ok = False
        with pytest.raises(RuntimeError):
            await server.load_adapter({"name": "policy", "path": adapter})
        with pytest.raises(RuntimeError, match="poisoned"):
            await _generate(server, request())
        assert not engine.calls

    asyncio.run(scenario())


def test_packed_modules_are_applied_under_their_packed_layer(tmp_path):
    """q/k/v, gate/up and the GDN in_proj pairs reach the served model only
    through its packed layers; the census names the layer that applies each."""

    async def scenario():
        server, engine = service()
        loaded = await server.load_adapter(
            {"name": "policy", "path": write_adapter(tmp_path / "adapter")}
        )
        assert loaded["lora_census"] == {
            "applied": APPLIED,
            "modules": {name: 2 if name in MLP else 1 for name in APPLIED},
        }
        assert loaded["adapter"] == server.adapter("policy").identity()
        # One read-only census before the one engine load.
        assert len(engine.census_calls) == 1 and len(engine.added) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "module",
    [
        # A suffix the served model has no LoRA layer for.
        "layers.1.self_attn.foo_proj",
        # A known suffix (the vision tower's qkv) outside the LoRA layers.
        "base_model.model.model.visual.blocks.0.attn.qkv",
        # A GDN projection on the full-attention layer.
        "layers.1.linear_attn.in_proj_b",
        # A layer the served model does not have.
        "layers.7.mlp.down_proj",
        # The MTP head, which vLLM skips while loading.
        "base_model.model.mtp.layers.0.mlp.down_proj",
    ],
)
def test_adapter_with_a_module_the_model_would_ignore_is_refused(tmp_path, module):
    """vLLM would load these adapters and never apply that module (or fail the
    load): the gateway refuses them before the engine sees the adapter."""

    async def scenario():
        server, engine = service()
        path = write_adapter(tmp_path / "adapter", [*LAYER_MODULES, module])
        with pytest.raises(ContractError, match="would not apply 1 of the adapter's 16"):
            await server.load_adapter({"name": "policy", "path": path})
        assert engine.added == [] and engine.controls == []
        assert server.failure is None and server.adapters == {}
        # The engine is untouched: a complete adapter loads next.
        loaded = await server.load_adapter(
            {"name": "policy", "path": write_adapter(tmp_path / "complete")}
        )
        assert loaded["adapter"]["id"] == 1 and len(engine.added) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"rank_pattern": {"q_proj": 4}}, "would drop"),
        ({"alpha_pattern": {"down_proj": 8}}, "would drop"),
    ],
)
def test_rank_or_alpha_patterns_vllm_would_drop_are_refused(tmp_path, config, match):
    async def scenario():
        server, engine = service()
        path = write_adapter(tmp_path / "adapter", **config)
        with pytest.raises(ContractError, match=match):
            await server.load_adapter({"name": "policy", "path": path})
        assert engine.census_calls == [] and engine.added == []
        assert server.failure is None

    asyncio.run(scenario())


def test_unreadable_adapter_is_refused_before_the_engine(tmp_path):
    async def scenario():
        server, engine = service()
        (tmp_path / "adapter_config.json").write_text("{}")
        with pytest.raises(ContractError, match="unreadable LoRA adapter"):
            await server.load_adapter({"name": "policy", "path": str(tmp_path)})
        assert engine.census_calls == [] and engine.added == []
        assert server.failure is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "result",
    [[], [{"mapped": []}], "census", [{"mapped": [None], "modules": [], "packed": {}}]],
)
def test_invalid_engine_census_poisons_before_any_load(tmp_path, result):
    async def scenario():
        server, engine = service()
        engine.census_result = result
        app = create_app(server)
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/adapters/load",
                json={"name": "policy", "path": write_adapter(tmp_path / "adapter")},
            )
            assert response.status == 503
            assert "invalid LoRA module census" in (await response.json())["error"]
        assert engine.added == [] and server.failure is not None

    asyncio.run(scenario())


def test_lora_engine_requires_the_census_worker_extension(monkeypatch):
    monkeypatch.setattr(
        "training.native_vllm_server.tokenizer_fingerprint", lambda tokenizer: {}
    )
    engine = Engine()
    engine.vllm_config.parallel_config = NS(worker_extension_cls="")
    with pytest.raises(RuntimeError, match="census worker extension"):
        NativeService(engine, sampling_factory=NS, lora_factory=NS)


def test_census_groups_factors_and_maps_through_the_engine_view():
    """The gateway's half of the census, against a canned engine view."""
    names = [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_B.weight",
    ]
    layer = "language_model.model.layers.0.self_attn"
    served = {
        "mapped": [f"{layer}.q_proj"] * 2 + [f"{layer}.o_proj"] * 2,
        "modules": [f"{layer}.qkv_proj", f"{layer}.o_proj"],
        "packed": {f"{layer}.qkv_proj": [f"{layer}.{n}" for n in ("q_proj", "k_proj", "v_proj")]},
    }
    assert lora_census(names, served) == {
        "applied": {"o_proj": "o_proj", "q_proj": "qkv_proj"},
        "modules": {"o_proj": 1, "q_proj": 1},
    }
    # The engine files the module's two factors under different names.
    with pytest.raises(ValueError, match="would not apply 1 of the adapter's 2"):
        lora_census(names, served | {"mapped": [f"{layer}.q_proj", None] + served["mapped"][2:]})
    with pytest.raises(ValueError, match="would not apply 2 of"):
        lora_census(names, served | {"modules": [], "packed": {}})


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (["m.q_proj.lora_A.weight"], "lack an A or B factor"),
        (["m.q_proj.lora_A.weight", "m.q_proj.lora_B.weight", "m.q_proj.weight"], "not a LoRA"),
        (["m.q_proj.lora_B.bias", "m.q_proj.lora_A.weight"], "not a LoRA"),
        (["m.e.lora_A.weight", "m.e.lora_embedding_A", "m.e.lora_B.weight"], "repeats"),
    ],
)
def test_census_refuses_tensors_that_are_not_complete_lora_pairs(names, match):
    with pytest.raises(ValueError, match=match):
        adapter_modules(names)


def test_census_reads_only_safetensors_adapters(tmp_path):
    pytest.importorskip("safetensors")
    path = tmp_path / "adapter"
    write_adapter(path, ["layers.1.self_attn.o_proj"])
    config, names = read_adapter(path)
    assert config["target_modules"] == ["o_proj"]
    assert sorted(names) == [
        "base_model.model.model.language_model.layers.1.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.1.self_attn.o_proj.lora_B.weight",
    ]
    (path / "adapter_model.safetensors").rename(path / "adapter_model.bin")
    with pytest.raises(ValueError, match="unreadable LoRA adapter"):
        read_adapter(path)


def test_engine_exception_never_retries():
    async def scenario():
        server, engine = service()
        engine.error = RuntimeError("worker died")
        with pytest.raises(RuntimeError, match="worker died"):
            await _generate(server, request())
        with pytest.raises(RuntimeError, match="poisoned"):
            await _generate(server, request(user="run#2"))
        assert len(engine.calls) == 1

    asyncio.run(scenario())


def test_http_native_routes_auth_metadata_warm_and_metrics(tmp_path):
    async def scenario():
        server, engine = service()
        adapter = write_adapter(tmp_path / "adapter")
        async with TestClient(TestServer(create_app(server, api_key="test"))) as client:
            assert (await client.get("/metadata")).status == 401
            client.session.headers["Authorization"] = "Bearer test"
            for path in [
                "/v1/models",
                "/v1/completions",
                "/v1/chat/completions",
                "/tokenize",
            ]:
                assert (await client.post(path)).status == 404
            loaded = await client.post(
                "/adapters/load", json={"name": "policy", "path": adapter}
            )
            assert loaded.status == 200
            metadata = await (
                await client.get("/metadata", params={"model": "policy"})
            ).json()
            assert (
                metadata["adapter"]["id"] == 1
                and metadata["model_revision"] == "pinned-model"
            )
            assert metadata["tokenizer_revision"] == "pinned-tokenizer"
            assert "secret" not in metadata
            warm = await client.post("/adapters/warm", json={"name": "policy"})
            warmed = await warm.json()
            assert warm.status == 200 and len(warmed["outputs"][0]["token_ids"]) == 1
            # The trainer confirms a load by this identity alone (no server log).
            assert warmed["adapter"] == (await loaded.json())["adapter"]
            # Beside the identity, the census of the layers applying the adapter.
            assert (await loaded.json())["lora_census"]["applied"] == APPLIED
            assert (await client.get("/health")).status == 200
            assert (await (await client.get("/runtime")).json())[
                "default_sampling_params"
            ] == {}
            models = await (await client.get("/models")).json()
            assert [model["name"] for model in models["models"]] == ["base", "policy"]
            assert (await client.get("/metrics")).status == 200
            generated = await client.post(
                "/generate",
                json=generation(
                    request(model="policy"), server.adapter("policy").identity()
                ),
            )
            assert (
                generated.status == 200
                and (await generated.json())["adapter"] == metadata["adapter"]
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changed", ["vocabulary", "merges", "normalizer", "decoder", "added_tokens"]
)
def test_tokenizer_identity_covers_more_than_template_and_special_ids(changed):
    pytest.importorskip(
        "ullm.backends.engine_proxy",
        reason="requires the private harness tokenizer fingerprint",
    )
    first = Tokenizer()
    payload = json.loads(first.backend_tokenizer.to_str())
    second = Tokenizer()
    if changed == "vocabulary":
        payload["model"]["vocab"] = {"a": 2, "b": 1}
    elif changed == "merges":
        payload["model"]["merges"] = ["b a"]
    elif changed == "normalizer":
        payload["normalizer"] = {"type": "NFC"}
    elif changed == "decoder":
        payload["decoder"] = {"type": "WordPiece"}
    else:
        payload["added_tokens"] = [{"id": 3, "content": "new"}]
    second.backend_tokenizer = NS(to_str=lambda: json.dumps(payload))
    assert first.chat_template == second.chat_template
    assert first.special_tokens_map == second.special_tokens_map
    assert tokenizer_fingerprint(first) != tokenizer_fingerprint(second)


def test_native_metadata_caches_canonical_complete_tokenizer_identity():
    native, _ = service(require_decoder=False)
    expected = tokenizer_fingerprint(Tokenizer())
    assert native.metadata()["tokenizer_fingerprint"] == expected
    payload = json.loads(Tokenizer().backend_tokenizer.to_str())
    equivalent = NS(
        backend_tokenizer=NS(
            to_str=lambda: json.dumps(payload, indent=4, sort_keys=True)
        )
    )
    assert tokenizer_fingerprint(equivalent) == expected
    native.metadata()["tokenizer_fingerprint"]["sha256"] = "tampered"
    assert native.metadata()["tokenizer_fingerprint"] == expected
    with pytest.raises(ContractError, match="serializable fast tokenizer"):
        tokenizer_fingerprint(NS())


@pytest.mark.parametrize(
    "requested",
    ["auto", "training.vllm_blackwell_worker.BlackwellGdnWorker", "unprotected.Worker"],
)
@pytest.mark.parametrize("version", ["0.25.1", "0.29.1.dev166+g3bda1ba09.precompiled"])
@pytest.mark.parametrize("head_k", [0, 32])
@pytest.mark.parametrize("lora", [False, True])
def test_main_passes_guarded_worker_to_engine_before_construction(
    monkeypatch, requested, head_k, version, lora
):
    import argparse
    import sys
    import training.native_vllm_server as server

    monkeypatch.setattr(server.importlib.metadata, "version", lambda _: version)
    engine_args = NS(
        worker_cls=requested,
        max_logprobs=20,
        enable_lora=lora,
        worker_extension_cls="",
    )
    engine_calls = []

    class Args:
        @staticmethod
        def add_cli_args(parser):
            return parser

        @staticmethod
        def from_cli_args(args):
            return engine_args

    class Engine:
        @staticmethod
        def from_engine_args(args):
            assert (
                args.worker_cls == "training.vllm_blackwell_worker.BlackwellGdnWorker"
            )
            # LoRA engines answer the gateway's census; Flash engines are untouched.
            assert args.worker_extension_cls == (WORKER_EXTENSION if lora else "")
            assert args.max_logprobs == max(20, head_k)
            engine_calls.append(args)
            raise RuntimeError("test engine constructed")

    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", NS(AsyncEngineArgs=Args))
    monkeypatch.setitem(
        sys.modules, "vllm.engine.async_llm_engine", NS(AsyncLLMEngine=Engine)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.argparse_utils",
        NS(FlexibleArgumentParser=argparse.ArgumentParser),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["native", "--port", "12345", "--score-centering-top-k", str(head_k)],
    )
    monkeypatch.setattr(server.web, "run_app", lambda app, **kwargs: asyncio.run(app))
    expected = (
        "guarded vLLM worker"
        if requested == "unprotected.Worker"
        else "test engine constructed"
    )
    with pytest.raises(RuntimeError, match=expected):
        server.main()
    assert len(engine_calls) == (0 if requested == "unprotected.Worker" else 1)


FORK = "0.29.1.dev1+g3bda1ba09.precompiled"


@pytest.mark.parametrize(
    ("argv", "expected", "version", "transfer", "policy_weights"),
    [
        ([], (), "0.25.1", None, False),
        (["--summary-forbidden-token-ids", "248058"], (248058,), "0.25.1", None, False),
        # 27B full weights keep their routes; on the Flash fork they carry merged policy weights.
        ([], (), "0.25.1", NS(backend="nccl"), False),
        ([], (), FORK, NS(backend="nccl"), True),
        ([], (), FORK, None, False),
    ],
)
def test_main_hands_the_summary_ban_and_publication_kind_to_the_service(
    monkeypatch, argv, expected, version, transfer, policy_weights
):
    import argparse
    import sys
    import training.native_vllm_server as server

    monkeypatch.setattr(server.importlib.metadata, "version", lambda _: version)
    engine = NS(shutdown=lambda: None)

    class Args:
        add_cli_args = staticmethod(lambda parser: parser)
        from_cli_args = staticmethod(
            lambda args: NS(worker_cls="auto", max_logprobs=20, enable_lora=False,
                            weight_transfer_config=transfer)
        )

    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", NS(AsyncEngineArgs=Args))
    monkeypatch.setitem(sys.modules, "vllm.engine.async_llm_engine",
                        NS(AsyncLLMEngine=NS(from_engine_args=lambda args: engine)))
    monkeypatch.setitem(sys.modules, "vllm.utils.argparse_utils",
                        NS(FlexibleArgumentParser=argparse.ArgumentParser))
    built = []

    def native_service(engine_, **kwargs):
        built.append((engine_, kwargs))
        raise RuntimeError("test service constructed")

    monkeypatch.setattr(server, "NativeService", native_service)
    monkeypatch.setattr(sys, "argv", ["native", "--port", "12345", *argv])
    monkeypatch.setattr(server.web, "run_app", lambda app, **kwargs: asyncio.run(app))
    with pytest.raises(RuntimeError, match="test service constructed"):
        server.main()
    assert built == [(engine, {"score_centering_top_k": 0, "summary_forbidden_token_ids": expected,
                               "policy_weights": policy_weights})]


def test_lora_engine_refuses_another_worker_extension():
    from training.native_vllm_server import configure_engine_worker

    engine_args = NS(worker_cls="auto", enable_lora=True, worker_extension_cls="x.Ext")
    with pytest.raises(RuntimeError, match="census worker extension"):
        configure_engine_worker(engine_args)
    flash = NS(worker_cls="auto", enable_lora=False, worker_extension_cls="x.Ext")
    configure_engine_worker(flash)
    assert flash.worker_extension_cls == "x.Ext"


@pytest.mark.parametrize("bias", [{2: -100}, {"2": -100.0}])
def test_native_validator_preserves_exact_logit_bias_for_sampling_params(bias):
    fields = validate_generation(
        request(logit_bias=bias), max_model_len=32, vocab_size=100
    )
    assert fields["logit_bias"] == {2: -100.0}
    assert type(next(iter(fields["logit_bias"]))) is int


@pytest.mark.parametrize(
    "bias",
    [
        {100: -100},
        {-1: -100},
        {True: -100},
        {1.0: -100},
        {2: "-100"},
        {2: float("nan")},
        {2: -101},
        [],
    ],
)
def test_native_validator_refuses_invalid_bias(bias):
    with pytest.raises(ContractError, match="logit_bias"):
        validate_generation(request(logit_bias=bias), max_model_len=32, vocab_size=100)


def test_native_generation_passes_logit_bias_to_engine():
    async def scenario():
        server, engine = service(summary_forbidden_token_ids=(2,))
        await _generate(server, request(logit_bias={"2": -100}))
        assert engine.calls[0][1].logit_bias == {2: -100.0}

    asyncio.run(scenario())


@pytest.mark.parametrize("verified", [(), (2,)])
def test_native_generation_biases_only_the_verified_summary_ban(verified):
    async def scenario():
        server, engine = service(summary_forbidden_token_ids=verified)
        with pytest.raises(ContractError, match="only the summary-forbidden token ids"):
            await _generate(server, request(logit_bias={"5": -100}))
        assert engine.calls == []

    asyncio.run(scenario())


def test_flash_runtime_attests_actual_ple_selector():
    from training.vllm_runtime import _validate_runtime_config, runtime_config

    engine = Engine()
    engine.vllm_config.model_config.architectures = ["Qwen4ExpForConditionalGeneration"]
    assert runtime_config(engine)["engram_cpu_offload"] is False  # no engram_config
    engine.vllm_config.engram_config = NS(cpu_offload=True)
    observed = runtime_config(engine)
    assert observed["engram_cpu_offload"] is True
    assert observed["model_architectures"] == ["Qwen4ExpForConditionalGeneration"]
    _validate_runtime_config(observed)
    engine.vllm_config.model_config.architectures = ["Qwen3VLForConditionalGeneration"]
    with pytest.raises(RuntimeError, match="engram_cpu_offload"):
        _validate_runtime_config(runtime_config(engine))
    engine.vllm_config.model_config.architectures = ["Qwen4ExpForConditionalGeneration"]
    offload = engine.vllm_config.offload_config
    offload.offload_backend = "uva"
    offload.uva.cpu_offload_gb = 48
    offload.uva.cpu_offload_params = {"ngram_embedding"}
    with pytest.raises(RuntimeError, match="cpu_offload_gb"):
        _validate_runtime_config(runtime_config(engine))


def _drafting(engine, **changes):
    """The fork's resolved SpeculativeConfig for the harness's drafting, edited."""
    from training.flash_drafting import FLASH_E307_DRAFTER

    engine.vllm_config.model_config.architectures = ["Qwen4ExpForConditionalGeneration"]
    engine.vllm_config.speculative_config = NS(**{
        "method": "mtp", "model": FLASH_E307_DRAFTER, "num_speculative_tokens": 4,
        "quantization": "modelopt_fp4", "draft_sample_method": "probabilistic",
        "gdn_recoverssm": True, "disable_eagle_block_drop": True,
        "rejection_sample_method": "standard", "accept_threshold_single": 1.0,
        "accept_threshold_acc": 1.0, "draft_top_k_top_p": False,
        # Resolved fields the attestation does not read.
        "draft_model_config": NS(model=FLASH_E307_DRAFTER), "parallel_drafting": False,
    } | changes)
    return engine


def test_flash_runtime_attests_the_drafter_the_config_asks_for():
    from training.config import SpeculativeDecodingConfig
    from training.vllm_runtime import _validate_runtime_config, runtime_config

    observed = json.loads(json.dumps(runtime_config(_drafting(Engine()))))
    expected = SpeculativeDecodingConfig(enabled=True).identity()
    assert observed["speculative_decoding"] is True
    assert observed["speculative"] == expected
    assert observed["model_architectures"] == ["Qwen4ExpForConditionalGeneration"]
    _validate_runtime_config(observed, speculative=expected)
    # A run that does not ask for drafting refuses a drafting engine, and the
    # converse; another drafter or draft count is another engine.
    with pytest.raises(RuntimeError, match="speculative_decoding must be False"):
        _validate_runtime_config(observed)
    undrafted = runtime_config(Engine())
    assert "speculative" not in undrafted
    with pytest.raises(RuntimeError, match="speculative_decoding must be True"):
        _validate_runtime_config(undrafted, speculative=expected)
    for other in ({"model": "/data/other-drafter"}, {"num_speculative_tokens": 3}):
        with pytest.raises(RuntimeError, match="vLLM runtime speculative must be"):
            _validate_runtime_config(observed, speculative=expected | other)


@pytest.mark.parametrize("change", [
    {"rejection_sample_method": "synthetic"},
    {"accept_threshold_single": 0.8},
    {"accept_threshold_acc": 0.5},
    {"draft_sample_method": "greedy"},
    {"method": "eagle3"},
    {"gdn_recoverssm": False},
    {"quantization": None},
    {"draft_top_k_top_p": True},
    {"num_speculative_tokens": 0},
])
def test_server_serves_only_the_harness_drafting_with_exact_verification(change):
    from training.native_vllm_server import _validate_served_runtime
    from training.vllm_runtime import runtime_config

    _validate_served_runtime(runtime_config(_drafting(Engine())), 0)
    with pytest.raises(RuntimeError, match="harness's MTP drafting with exact verification"):
        _validate_served_runtime(runtime_config(_drafting(Engine(), **change)), 0)


def test_server_refuses_drafting_on_another_architecture():
    from training.native_vllm_server import _validate_served_runtime
    from training.vllm_runtime import runtime_config

    engine = _drafting(Engine())
    engine.vllm_config.model_config.architectures = ["Qwen3VLForConditionalGeneration"]
    with pytest.raises(RuntimeError, match="on Qwen4Exp"):
        _validate_served_runtime(runtime_config(engine), 0)


def test_native_service_starts_with_the_harness_drafting_only():
    service(_drafting(Engine()))
    with pytest.raises(RuntimeError, match="exact verification"):
        service(_drafting(Engine(), accept_threshold_single=0.8))


def _emit_logprob(engine, value):
    """Return every sampled token with the given raw logprob."""

    def replace_output(output):
        completion = output.outputs[0]
        completion.logprobs = [
            {token: NS(rank=None, logprob=value)} for token in completion.token_ids
        ]
        return output

    engine.output_edit = replace_output


def test_sampled_logprob_rounding_above_zero_is_clamped_not_fatal():
    async def scenario():
        server, engine = service()
        # A greedy or near-certain draw can round a few 1e-4 above zero in the
        # engine's log-softmax; the bound is 0, and the engine stays healthy.
        _emit_logprob(engine, 5e-4)
        result = await _generate(server, request())
        assert result["output_logprobs"] == [0.0, 0.0]
        assert server.failure is None
        second = await _generate(server, request(user="run#2"))
        assert second["output_logprobs"] == [0.0, 0.0]

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0.5, float("nan"), float("inf")])
def test_sampled_logprob_beyond_rounding_poisons_the_engine(value):
    async def scenario():
        server, engine = service()
        _emit_logprob(engine, value)
        with pytest.raises(RuntimeError, match="invalid raw logprob"):
            await _generate(server, request())
        assert server.failure is not None and "invalid raw logprob" in server.failure

    asyncio.run(scenario())


def test_sampled_logprob_clamp_is_reported_once_per_request(caplog):
    async def scenario():
        server, engine = service()
        _emit_logprob(engine, 5e-4)
        with caplog.at_level("WARNING", logger="training.native_vllm_server"):
            await _generate(server, request())
        (message,) = [
            r.getMessage() for r in caplog.records if "clamped" in r.getMessage()
        ]
        assert (
            "clamped 2 sampled logprob(s) in (0, 0.001] to 0 in request run#1"
            in message
        )
        assert "largest 0.0005 for token 4" in message
        caplog.clear()
        _emit_logprob(engine, -0.4)
        with caplog.at_level("WARNING", logger="training.native_vllm_server"):
            await _generate(server, request(user="run#2"))
        assert not [r for r in caplog.records if "clamped" in r.getMessage()]

    asyncio.run(scenario())


class PolicyEngine(Engine):
    """A Flash engine: no LoRA, NCCL weight transfer, workers that receive merged policy weights.

    Each worker is a PolicyWeightReceiver over its own packed served model,
    called by the method names the guarded worker exposes; ``sent`` is what
    the learner broadcasts, in order (each worker receives its own copy).
    """

    def __init__(self, checkpoint_dir, checkpoint, workers=1, **faults):
        super().__init__()
        from policy_weight_fixtures import ServedModel
        from training.flash_policy_weights import PolicyWeightReceiver

        self.vllm_config.lora_config = None
        self.vllm_config.weight_transfer_config = NS(backend="nccl")
        self.vllm_config.parallel_config = NS(world_size_across_dp=workers, worker_extension_cls="")
        self.models = [ServedModel(checkpoint, **faults) for _ in range(workers)]
        self.receivers = [PolicyWeightReceiver(checkpoint_dir) for _ in range(workers)]
        self.sent = []
        self.result_changes = {}
        self.paused = False

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.controls.append(("rpc", method))
        results = []
        for index, (receiver, model) in enumerate(zip(self.receivers, self.models, strict=True)):
            if method == "prepare_flash_policy_weights":
                result = receiver.prepare(model, *args)
            elif method == "receive_flash_policy_weights":
                feed = iter(self.sent)
                result = receiver.receive(model, *args, lambda buffer, feed=feed: buffer.copy_(next(feed)))
            else:
                assert method == "verify_flash_policy_weights"
                result = receiver.verify(*args) | self.result_changes.get(index, {})
            results.append(result)
        return results

    async def init_weight_transfer_engine(self, request):
        self.controls.append(("init", request.init_info))

    async def pause_generation(self, *, mode, clear_cache):
        self.controls.append(("pause", (mode, clear_cache)))
        self.paused = True

    async def is_paused(self):
        return self.paused

    async def resume_generation(self):
        self.controls.append(("resume", None))
        self.paused = False

    async def start_weight_update(self):
        raise AssertionError("Flash publications never use vLLM's layerwise reload")

    update_weights = finish_weight_update = start_weight_update


INIT = {"init_info": {"master_address": "learner", "master_port": 12345, "rank_offset": 1, "world_size": 3}}


@pytest.fixture(autouse=True)
def _weight_requests(monkeypatch):
    """The pinned engine's typed requests, faked: a namespace with the info."""
    monkeypatch.setattr(
        "training.native_vllm_server._weight_request", lambda kind, info: NS(**{f"{kind}_info": info})
    )


def _policy_setup(tmp_path, workers=1, **faults):
    from policy_weight_fixtures import write_checkpoint

    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    engine = PolicyEngine(tmp_path / "checkpoint", checkpoint, workers=workers, **faults)
    return engine, checkpoint


def _publication(engine, checkpoint, **kwargs):
    """A publication's start body, its update body and tensors; the engine's wire carries them."""
    from policy_weight_fixtures import make_publication
    from training.flash_policy_weights import manifest_sha256
    from training.policy_sync.vllm_nccl import WeightManifest

    manifest, tensors = make_publication(checkpoint, **kwargs)
    engine.sent = [tensors[name] for name in manifest["names"]]
    label = {"kind": "policy_weights", "policy_sha256": manifest_sha256(manifest),
             "policy_iteration": manifest["iteration"]}
    update = WeightManifest(tuple(manifest["names"]), ("bfloat16",) * len(manifest["names"]),
                            tuple(tuple(shape) for shape in manifest["shapes"])).update_info(packed=False)
    return {**label, "manifest": manifest}, {"update_info": update}, tensors


async def _publish(server, start, update):
    for operation, body in (("init", INIT), ("pause", {}), ("start", start), ("update", update),
                            ("finish", {}), ("resume", {})):
        if operation == "init" and server._weight_phase is not None:
            continue
        result = await getattr(server, f"{operation}_weights")(body)
    return result


def test_flash_publication_over_the_nccl_routes_is_verified_and_tags_every_draw(tmp_path):
    from training.policy_identity import policy_weights_identity

    async def scenario():
        engine, checkpoint = _policy_setup(tmp_path, workers=2)
        server, _ = service(engine, require_decoder=False, policy_weights=True)
        addresses = [p.data_ptr() for p in engine.models[0].parameters()]
        start, update, tensors = _publication(engine, checkpoint, iteration=4, routers=True)
        label = {key: value for key, value in start.items() if key != "manifest"}
        async with TestClient(TestServer(create_app(server, api_key="test"))) as client:
            client.session.headers["Authorization"] = "Bearer test"
            info = await (await client.get("/policy_weights/info")).json()
            # Until a publication the server serves the checkpoint's weights.
            assert info == {"sha256": None, "iteration": None, "tensors": 0,
                            "server_instance_id": server.instance_id}
            assert policy_weights_identity(server.adapter("base").identity()) is None
            assert (await (await client.get("/weights/info")).json())["world_size"] == 2
            for operation, body in (("init", INIT), ("pause", {}), ("start", start), ("update", update),
                                    ("finish", {})):
                response = await client.post(f"/weights/{operation}", json=body)
                assert response.status == 200, await response.text()
            # Verified, not yet live: draws still carry no publication.
            assert policy_weights_identity(server.adapter("base").identity()) is None
            response = await client.post("/weights/resume", json={})
            assert response.status == 200
            resumed = (await response.json())["adapter"]
            info = await (await client.get("/policy_weights/info")).json()
            assert (info["sha256"], info["iteration"], info["tensors"]) == (start["policy_sha256"], 4, len(tensors))
            metadata = await (await client.get("/metadata", params={"model": "base"})).json()
            # The file route is gone.
            assert (await client.post("/policy_weights/apply", json={})).status in (404, 405)
        # Every base draw now names the publication it was served with.
        assert metadata["adapter"] == resumed == server.adapter("base").identity() == {
            "name": "base", "id": 0, "path": None, "server_instance_id": server.instance_id, **label,
        }
        for model in engine.models:
            for name, value in tensors.items():
                assert torch_equal(model.served(name), value)
        # In place: CUDA graphs keep reading the same storage.
        assert [p.data_ptr() for p in engine.models[0].parameters()] == addresses
        assert engine.controls == [
            ("init", INIT["init_info"]), ("pause", ("wait", True)), ("rpc", "prepare_flash_policy_weights"),
            ("rpc", "receive_flash_policy_weights"), ("rpc", "verify_flash_policy_weights"), ("reset", None),
            ("resume", None),
        ]
        assert server.failure is None and not server.mutating
        # The next publication goes through the same initialized group.
        start, update, tensors = _publication(engine, checkpoint, iteration=5, routers=True, seed=3)
        assert (await _publish(server, start, update))["adapter"]["policy_iteration"] == 5
        assert all(torch_equal(engine.models[1].served(name), value) for name, value in tensors.items())

    asyncio.run(scenario())


def torch_equal(a, b):
    import torch

    return torch.equal(a, b)


@pytest.mark.parametrize(
    "change", ["no-manifest", "hash", "iteration", "full-label", "extra-key", "order", "old-format"],
)
def test_flash_start_rejections_never_reach_the_engine(tmp_path, change):
    async def scenario():
        engine, checkpoint = _policy_setup(tmp_path)
        server, _ = service(engine, require_decoder=False, policy_weights=True)
        start, _, _ = _publication(engine, checkpoint)
        manifest = start["manifest"]
        body = {
            "no-manifest": {key: value for key, value in start.items() if key != "manifest"},
            "hash": {**start, "policy_sha256": "0" * 64},
            "iteration": {**start, "policy_iteration": manifest["iteration"] + 1},
            "full-label": {"kind": "full_weight", "weight_version": "a" * 32, "manifest_sha256": "b" * 64,
                           "manifest": manifest},
            "extra-key": {**start, "path": "/data/policy_weights.safetensors"},
            "order": {**start, "manifest": {**manifest, "names": manifest["names"][::-1]}},
            "old-format": {**start, "manifest": {**manifest, "format": "flash_policy_weights/2"}},
        }[change]
        await server.init_weights(INIT)
        await server.pause_weights({})
        with pytest.raises(ContractError):
            await server.start_weights(body)
        assert ("rpc", "prepare_flash_policy_weights") not in engine.controls and server.failure is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault",
    ["extra", "missing", "shape", "update", "misplaced", "checksum", "wire", "worker_disagreement",
     "device_report", "cache_reset"],
)
def test_flash_publication_refusal_or_mismatch_poisons_the_engine(tmp_path, fault):
    from policy_weight_fixtures import HIDDEN

    name = "model.language_model.layers.0.self_attn.q_proj.weight"

    async def scenario():
        engine, checkpoint = _policy_setup(tmp_path, workers=2, misplace={name} if fault == "misplaced" else ())
        kwargs = {
            "extra": {"extra": {"model.language_model.layers.2.mlp.shared_expert.down_proj.weight":
                                torch_zeros(HIDDEN, 5)}},
            "missing": {"drop": {name}},
            "shape": {"reshape": "model.language_model.layers.1.linear_attn.out_proj.weight"},
            # The learner's checksum of a tensor differs from what it sends.
            "checksum": {"checksum_offset": name},
        }.get(fault, {})
        start, update, tensors = _publication(engine, checkpoint, **kwargs)
        if fault == "wire":
            index = start["manifest"]["names"].index(name)
            engine.sent[index] = engine.sent[index].flip(1).contiguous()
        elif fault == "update":
            update = {"update_info": {**update["update_info"], "names": update["update_info"]["names"][::-1]}}
        elif fault == "worker_disagreement":
            engine.result_changes = {1: {"iteration": 99}}
        elif fault == "device_report":
            engine.result_changes = {0: {"sha256": "0" * 64}, 1: {"sha256": "0" * 64}}
        elif fault == "cache_reset":
            engine.reset_ok = False
        server, _ = service(engine, require_decoder=False, policy_weights=True)
        with pytest.raises((RuntimeError, ValueError)):
            await _publish(server, start, update)
        assert server.failure is not None and server.mutating
        assert server._policy_weights is None
        assert "policy_sha256" not in server.adapter("base").identity()
        assert ("resume", None) not in engine.controls
        with pytest.raises(RuntimeError, match="poisoned"):
            await server.policy_weights_info()
        async with TestClient(TestServer(create_app(server))) as client:
            assert (await client.get("/health")).status == 503

    asyncio.run(scenario())


def torch_zeros(*shape):
    import torch

    return torch.zeros(*shape).bfloat16()


def test_flash_publication_requires_a_drained_engine_and_info_does_not(tmp_path):
    async def scenario():
        engine, checkpoint = _policy_setup(tmp_path)
        server, _ = service(engine, policy_weights=True)
        await server.init_weights(INIT)
        engine.release = asyncio.Event()
        task = asyncio.create_task(_generate(server, request()))
        await engine.entered.wait()
        with pytest.raises(ContractError, match="drained"):
            await server.pause_weights({})
        # The window-start check is a read: it never waits for draws to finish.
        assert (await server.policy_weights_info())["sha256"] is None
        engine.release.set()
        await task
        assert ("pause", ("wait", True)) not in engine.controls and server.failure is None
        server.mutating = True
        with pytest.raises(ContractError, match="during a control update"):
            await server.policy_weights_info()

    asyncio.run(scenario())


def test_flash_engine_refuses_adapter_loads_but_serves_warm_ups(tmp_path):
    async def scenario():
        engine, checkpoint = _policy_setup(tmp_path)
        server, _ = service(engine, policy_weights=True)
        with pytest.raises(ContractError, match="without LoRA"):
            await server.load_adapter({"name": "policy", "path": str(tmp_path)})
        start, update, _ = _publication(engine, checkpoint, iteration=2)
        await _publish(server, start, update)
        warm = await server.warm_adapter({"name": "base"})
        assert warm.adapter["policy_sha256"] == start["policy_sha256"]
        assert warm.adapter["policy_iteration"] == 2
        assert engine.calls[-1][3] is None  # no LoRARequest
        assert engine.added == [] and server.failure is None

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["no-transfer", "lora", "ipc"])
def test_flash_policy_weights_require_nccl_without_lora(tmp_path, change):
    engine, _ = _policy_setup(tmp_path)
    if change == "no-transfer":
        engine.vllm_config.weight_transfer_config = None
    elif change == "lora":
        engine.vllm_config.lora_config = NS(max_lora_rank=16)
    else:
        engine.vllm_config.weight_transfer_config = NS(backend="ipc")
    with pytest.raises(RuntimeError, match="NCCL weight transfer without LoRA"):
        service(engine, require_decoder=False, policy_weights=True)
    # Without policy weights an engine has no merged-weights routes.
    engine, _ = _policy_setup(tmp_path / "other")
    engine.vllm_config.weight_transfer_config = None
    server, _ = service(engine, require_decoder=False)
    with pytest.raises(ContractError, match="does not serve Flash policy weights"):
        asyncio.run(server.policy_weights_info())
