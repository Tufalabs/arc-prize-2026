"""Native AsyncLLMEngine service: integer prompts, exact raw sampled logprobs.

The trainer gateway admits rollout draws. This process owns SamplingParams,
LoRARequest, full-weight and merged policy weight identities. There is no
chat/completion API, rendering, or retry.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hmac
import hashlib
import importlib.metadata
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import time
import uuid

from aiohttp import web

from training.engine_protocol import unpack_generation
from training.vllm_runtime import (
    TRANSPORT,
    _validate_runtime_config,
    is_flash_vllm_version,
    runtime_config,
)

log = logging.getLogger(__name__)

# Sampled-token logprobs above zero by at most this much are rounding, clamped to 0.
_LOGPROB_ROUNDING_TOLERANCE = 1e-3

# Request-local evidence avoids retaining window-sized heads in the server.
_response_head = ContextVar("response_head", default=None)

# What every summary-forbidden token must decode to under the served tokenizer:
# the Qwen tool-call marker, which a summary must never open.
SUMMARY_FORBIDDEN_TEXT = "<tool_call>"


class ContractError(ValueError):
    """Request rejected before engine dispatch."""


def _validate_served_runtime(config: dict, score_centering_top_k: int) -> None:
    """The server serves any qualified drafter it was started with (the
    harness's MTP drafting, exact verification); the trainer pins which one
    against its config (validate_server_runtime)."""
    _validate_runtime_config(
        config,
        score_centering_top_k=score_centering_top_k,
        speculative=config.get("speculative"),
    )


def _bounded_logprob(value, token: int, clamped: list) -> float:
    """A raw engine logprob; rounding a few 1e-4 above zero is clamped to 0."""
    value = float(value)
    if not math.isfinite(value) or value > _LOGPROB_ROUNDING_TOLERANCE:
        raise RuntimeError(
            f"native engine returned an invalid raw logprob: {value!r} for token {token}"
        )
    if value > 0:
        clamped.append((value, token))
        return 0.0
    return value


def _warn_clamped(kind: str, request_id: str, clamped: list) -> None:
    if clamped:
        value, token = max(clamped)
        log.warning(
            "clamped %d %s logprob(s) in (0, %g] to 0 in request %s; largest %r for token %d",
            len(clamped),
            kind,
            _LOGPROB_ROUNDING_TOLERANCE,
            request_id,
            value,
            token,
        )


def validate_bind_auth(host: str, api_key: str | None) -> None:
    """Require authentication before exposing native serving off loopback."""
    if api_key and api_key != api_key.strip():
        raise RuntimeError("SERVER_API_KEY must not contain surrounding whitespace")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower().rstrip(".") == "localhost"
    if not loopback and (not api_key or not api_key.strip()):
        raise RuntimeError(
            "Non-loopback native serving requires a nonempty SERVER_API_KEY"
        )


def tokenizer_fingerprint(tokenizer) -> dict[str, str]:
    """Require the shared tokenizer identity before serving any token IDs."""
    from ullm.backends.engine_proxy import tokenizer_fingerprint as fingerprint

    try:
        return fingerprint(tokenizer)
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError(
            "native serving requires a serializable fast tokenizer"
        ) from error


@dataclass(frozen=True)
class Adapter:
    name: str
    id: int
    path: str | None
    server_instance_id: str
    request: object = None
    publication: dict | None = None

    def identity(self) -> dict:
        return {
            key: getattr(self, key)
            for key in ("name", "id", "path", "server_instance_id")
        } | (self.publication or {})


def _weight_request(kind: str, info: dict):
    """Construct the pinned engine's typed request at the native boundary."""
    from vllm.distributed.weight_transfer.base import (
        WeightTransferInitRequest,
        WeightTransferUpdateRequest,
    )

    if kind == "init":
        return WeightTransferInitRequest(init_info=info)
    return WeightTransferUpdateRequest(update_info=info)


def validate_generation(
    body: dict, *, max_model_len: int, vocab_size: int, score_centering_top_k: int = 0
) -> dict:
    """Only native sampling fields, including bounded explicit token biases."""
    if not isinstance(body, dict):
        raise ContractError("generation body must be an object")
    required = {
        "model",
        "user",
        "prompt",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "min_tokens",
        "logprobs",
        "skip_special_tokens",
    }
    allowed = required | {"seed", "ignore_eos", "n", "logit_bias"}
    if set(body) - allowed or required - set(body):
        raise ContractError(
            f"native fields missing={sorted(required - set(body))}, unknown={sorted(set(body) - allowed)}"
        )
    if not isinstance(body["model"], str) or not body["model"]:
        raise ContractError("model must be a nonempty name")
    stamp = body["user"]
    if (
        not isinstance(stamp, str)
        or not all(stamp.rpartition("#")[::2])
        or "#" not in stamp
    ):
        raise ContractError("user must be a run_id#nonce stamp")
    prompt = body["prompt"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(
            type(token) is not int or not 0 <= token < vocab_size for token in prompt
        )
    ):
        raise ContractError("prompt must be a nonempty list of vocabulary token IDs")
    maximum = body["max_tokens"]
    if (
        type(maximum) is not int
        or maximum <= 0
        or len(prompt) + maximum > max_model_len
    ):
        raise ContractError(
            "prompt plus positive max_tokens must fit the engine context"
        )
    for key, expected in {
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_p": 0.0,
        "min_tokens": 0,
    }.items():
        value = body[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value != expected
        ):
            raise ContractError(f"{key} must explicitly be {expected}")
    if type(body["min_tokens"]) is not int:
        raise ContractError("min_tokens must be an integer")
    if type(body["logprobs"]) is not int or body["logprobs"] not in (
        {0, score_centering_top_k} if score_centering_top_k else {0}
    ):
        raise ContractError(
            f"logprobs must be 0 (evaluation) or configured k={score_centering_top_k}"
        )
    if (
        body["skip_special_tokens"] is not False
        or body.get("ignore_eos", False) is not False
    ):
        raise ContractError("skip_special_tokens and ignore_eos must be false")
    for key in ("temperature", "top_p"):
        value = body[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ContractError(f"invalid {key}")
        if key == "top_p" and not 0 < value <= 1:
            raise ContractError("top_p must be in (0, 1]")
    if type(body["top_k"]) is not int or (
        body["top_k"] not in (0, -1) and body["top_k"] <= 0
    ):
        raise ContractError("top_k must be disabled or a positive integer")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise ContractError("n must be 1")
    if "seed" in body and type(body["seed"]) is not int:
        raise ContractError("seed must be an integer")
    bias = {}
    if "logit_bias" in body:
        from training.engine_protocol import EngineProtocolError, normalize_logit_bias

        try:
            bias = normalize_logit_bias(body["logit_bias"])
        except EngineProtocolError as error:
            raise ContractError(str(error)) from error
        if any(token >= vocab_size for token in bias):
            raise ContractError("logit_bias token IDs must be in the model vocabulary")
    return {
        "n": 1,
        **({"logit_bias": bias} if bias else {}),
        **{
            key: body[key]
            for key in (
                "temperature",
                "top_p",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "min_p",
                "min_tokens",
            )
        },
        "top_k": body["top_k"] if body["top_k"] > 0 else -1,
        "seed": body.get("seed"),
        "logprobs": body["logprobs"],
        "skip_special_tokens": False,
        "ignore_eos": False,
        "detokenize": True,
    }


def image_pixel_ceiling(processor, mm_processor_kwargs: dict) -> int:
    """The pixel count above which the served image processor shrinks an image.

    Qwen's resize rule (``smart_resize``) scales an image down when its pixel
    count, each side rounded to the patch grid, exceeds ``max_pixels``: the
    engine's ``mm_processor_kwargs`` value when it names one, else the image
    processor's ``size["longest_edge"]`` (the checkpoint's preprocessor
    config; the processor is built with the same kwargs). The launcher names
    none, so Flash serves its checkpoint's 16,777,216. A processor that
    states neither is refused: no worker could check that its images arrive
    unshrunk.
    """
    ceiling = mm_processor_kwargs.get("max_pixels")
    if ceiling is None:
        try:
            ceiling = processor.image_processor.size["longest_edge"]
        except (AttributeError, KeyError, TypeError):
            ceiling = None
    if type(ceiling) is not int or ceiling <= 0:
        raise ContractError(
            "native image serving requires a positive integer pixel ceiling "
            "(mm_processor_kwargs max_pixels or the image processor's "
            f"size['longest_edge']); got {ceiling!r}"
        )
    return ceiling


def summary_forbidden_token_ids(value: str) -> tuple[int, ...]:
    """``--summary-forbidden-token-ids``: comma-separated token ids, or none."""
    ids = tuple(int(item) for item in value.split(",")) if value.strip() else ()
    if any(token < 0 for token in ids) or len(set(ids)) != len(ids):
        raise ValueError("summary-forbidden token ids must be distinct and nonnegative")
    return ids


def verify_summary_forbidden_tokens(tokenizer, token_ids, vocab_size: int) -> frozenset:
    """The summary's banned ids, each the tool-call marker under the served tokenizer.

    The ids reach the server only as the gateway's over-trigger ``logit_bias``,
    copied as literals from the Duck (``QWEN_TOOL_CALL_TOKEN_ID``) into the
    recipe, so a tokenizer that put another token there would ban the wrong one
    in silence. The launcher passes the static id; startup decodes it.
    """
    for token in token_ids:
        if type(token) is not int or not 0 <= token < vocab_size:
            raise RuntimeError(
                f"summary-forbidden token id {token!r} is outside the model vocabulary"
            )
        text = tokenizer.decode([token], skip_special_tokens=False)
        if text != SUMMARY_FORBIDDEN_TEXT:
            raise RuntimeError(
                f"summary-forbidden token id {token} decodes to {text!r} under the served "
                f"tokenizer, not {SUMMARY_FORBIDDEN_TEXT!r}; set "
                "SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS to this tokenizer's tool-call id"
            )
    return frozenset(token_ids)


def _resolved_architecture(model_config) -> str:
    """The engine's single resolved model architecture; ullm binds image identity to it."""
    architectures = list(
        getattr(model_config, "architectures", None)
        or getattr(getattr(model_config, "hf_config", None), "architectures", None)
        or []
    )
    if len(architectures) != 1:
        raise ContractError(
            f"native engine must resolve exactly one model architecture, got {architectures}"
        )
    return str(architectures[0])


class NativeService:
    def __init__(
        self,
        engine,
        *,
        sampling_factory=None,
        lora_factory=None,
        score_centering_top_k=0,
        summary_forbidden_token_ids=(),
        policy_weights=False,
    ):
        # Lazy imports keep contract/HTTP tests independent of CUDA and vLLM.
        if sampling_factory is None:
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind

            def sampling_factory(**kwargs):
                return SamplingParams(
                    **kwargs, output_kind=RequestOutputKind.FINAL_ONLY
                )

        if lora_factory is None:
            from vllm.lora.request import LoRARequest

            lora_factory = LoRARequest
        self.score_centering_top_k = score_centering_top_k
        self.engine = engine
        self.sampling_factory = sampling_factory
        self.lora_factory = lora_factory
        self.instance_id = uuid.uuid4().hex
        self.tokenizer = engine.get_tokenizer()
        self._tokenizer_fingerprint = tokenizer_fingerprint(self.tokenizer)
        self.model = engine.vllm_config.model_config
        transfer_config = getattr(engine.vllm_config, "weight_transfer_config", None)
        # Flash engines receive merged policy weights over the same NCCL
        # routes (training/flash_policy_weights.py); others there take full
        # BF16 weights.
        self.policy_weights = bool(policy_weights)
        if self.policy_weights and (
            transfer_config is None
            or transfer_config.backend != "nccl"
            or getattr(engine.vllm_config, "lora_config", None) is not None
        ):
            raise RuntimeError(
                "Flash policy-weights serving requires NCCL weight transfer without LoRA"
            )
        self.full_weight = transfer_config is not None and not self.policy_weights
        if self.full_weight and (
            transfer_config.backend != "nccl"
            or getattr(engine.vllm_config, "lora_config", None) is not None
            or str(self.model.dtype) != "torch.bfloat16"
            or getattr(self.model, "quantization", None) is not None
        ):
            raise RuntimeError(
                "Full-weight native serving requires unquantized BF16 NCCL without LoRA"
            )
        if self._lora_enabled():
            from training.lora_census import WORKER_EXTENSION

            # Adapter loads take the census through this worker extension.
            if engine.vllm_config.parallel_config.worker_extension_cls != WORKER_EXTENSION:
                raise RuntimeError(
                    "native LoRA serving requires the LoRA census worker extension "
                    f"{WORKER_EXTENSION} (configure_engine_worker)"
                )
        self._weight_phase = None
        self._weight_control_active = False
        self._weight_publication = None
        self._pending_weight_publication = None
        # The pending Flash publication's manifest, between start and resume.
        self._pending_policy_manifest = None
        self._weight_versions: set[str] = set()
        self._media_processor = None
        self._processing_identity = None
        self._image_max_pixels = None
        self.max_model_len = self.model.max_model_len
        self.base_name = self.model.served_model_name
        # Qwen can have model logit rows absent from the tokenizer. Preserve
        # those sampled IDs, including when they return in the next prompt.
        # Tokenizers may also expose sparse or multimodal-only input IDs:
        # match vLLM's input bound while output IDs stay within model logits.
        self.output_vocab_size = self.model.get_vocab_size()
        self.input_vocab_size = max(
            self.output_vocab_size, self.tokenizer.max_token_id + 1
        )
        # The only ids a request may bias: the summary's ban, checked here.
        self.summary_forbidden_token_ids = verify_summary_forbidden_tokens(
            self.tokenizer, summary_forbidden_token_ids, self.output_vocab_size
        )
        self.adapters: dict[str, Adapter] = {}
        self.next_adapter_id = 1
        # The merged policy weights last published over /weights/* (None: the
        # checkpoint's); training/flash_policy_weights.py.
        self._policy_weights: dict | None = None
        self.inflight: set[str] = set()
        self.mutating = False
        self.failure: str | None = None
        self.lock = asyncio.Lock()
        _validate_served_runtime(runtime_config(engine), score_centering_top_k)
        # Readiness includes cold processor loading; metadata is a short
        # control request and must not perform this work after admission.
        multimodal = getattr(self.model, "multimodal_config", None)
        if multimodal is not None and multimodal.get_limit_per_prompt("image") > 0:
            log.info("loading verified image processor before accepting requests")
            self._ensure_media_processor()
            log.info("verified image processor ready")

    def _healthy(self) -> None:
        if self.failure is not None:
            raise RuntimeError(f"native engine is poisoned: {self.failure}")

    def poison(self, error: BaseException) -> None:
        if self.failure is None:
            self.failure = f"{type(error).__name__}: {error}"
            log.error("FATAL native engine failure: %s", self.failure)

    def adapter(self, name: str) -> Adapter:
        if name == self.base_name:
            return Adapter(
                name, 0, None, self.instance_id, publication=self._weight_publication
            )
        if name not in self.adapters:
            raise ContractError(f"unknown model alias {name!r}")
        return self.adapters[name]

    def metadata(self, name: str | None = None) -> dict:
        multimodal = getattr(self.model, "multimodal_config", None)
        if multimodal is not None and multimodal.get_limit_per_prompt("image") > 0:
            self._ensure_media_processor()
        return {
            "transport": TRANSPORT,
            "vllm_version": importlib.metadata.version("vllm"),
            "model_id": self.model.model,
            "model_revision": self.model.revision,
            "served_model_name": self.base_name,
            "tokenizer_id": self.model.tokenizer,
            "tokenizer_revision": self.model.tokenizer_revision,
            "tokenizer_fingerprint": dict(self._tokenizer_fingerprint),
            "max_model_len": self.max_model_len,
            "chat_template": self.tokenizer.chat_template,
            "special_tokens_map": self.tokenizer.special_tokens_map,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "adapter": self.adapter(name or self.base_name).identity(),
            # Above this many pixels the served processor shrinks an image
            # (None: images disabled); workers check their largest render fits.
            "image_max_pixels": self._image_max_pixels,
            **(
                {
                    "processing": self._processing_identity,
                    "mm_processor_kwargs": dict(multimodal.mm_processor_kwargs or {}),
                }
                if self._processing_identity is not None
                else {}
            ),
        }

    def _ensure_media_processor(self) -> None:
        """Use the engine's pinned processor, without loading GPU model weights."""
        if self._media_processor is not None:
            return
        from ullm.media import processing_identity

        multimodal = self.model.multimodal_config
        if multimodal.get_limit_per_prompt("image") <= 0:
            raise ContractError("native engine has images disabled")
        kwargs = dict(multimodal.mm_processor_kwargs or {})
        processor = self.engine.renderer.mm_processor.info.get_hf_processor(**kwargs)
        ceiling = image_pixel_ceiling(processor, kwargs)
        identity = processing_identity(
            processor,
            model=self.model.model,
            revision=self.model.revision,
            mm_processor_kwargs=kwargs,
            model_architecture=_resolved_architecture(self.model),
        )
        self._media_processor, self._processing_identity = processor, identity
        self._image_max_pixels = ceiling

    def _multimodal_input(self, prompt: list[int], media: dict):
        """Build a real expanded EngineInput and verify its tensors and M-RoPE."""
        from ullm.media import reproduce_media, tensor_evidence
        from vllm.inputs.engine import mm_input
        from vllm.multimodal.inputs import (
            MultiModalFeatureSpec,
            MultiModalKwargsItems,
            PlaceholderRange,
        )

        if _resolved_architecture(self.model) == "Qwen4ExpForConditionalGeneration":
            # The pinned fork inherits Qwen3-VL's image position algorithm;
            # select its actual class so an override cannot go unnoticed.
            from vllm.models.qwen4_exp.nvidia.model import (
                Qwen4ExpForConditionalGeneration as position_model,
            )
        else:
            from vllm.model_executor.models.qwen3_vl import (
                Qwen3VLForConditionalGeneration as position_model,
            )

        self._ensure_media_processor()
        tensors, evidence = reproduce_media(
            prompt, media, self._media_processor, self._processing_identity
        )
        fields = self.engine.renderer.mm_processor._get_mm_fields_config(
            tensors, self._processing_identity["mm_processor_kwargs"]
        )
        kwargs = MultiModalKwargsItems.from_hf_inputs(tensors, fields)
        consumed = kwargs.get_data()
        if set(consumed) != set(tensors) or any(
            tensor_evidence(consumed[key]) != evidence["tensors"][key]
            for key in tensors
        ):
            raise ContractError(
                "vLLM multimodal field conversion changed image tensors"
            )
        ranges = [
            PlaceholderRange(offset=item["offset"], length=item["length"])
            for item in evidence["layout"]
        ]
        hashes = [
            hashlib.sha256(
                json.dumps([self._processing_identity, value], sort_keys=True).encode()
            ).hexdigest()
            for value in evidence["image_hashes"]
        ]
        features = [
            MultiModalFeatureSpec(
                data=data, modality="image", identifier=key, mm_position=span
            )
            for data, key, span in zip(kwargs["image"], hashes, ranges, strict=True)
        ]
        positions, _ = position_model._get_mrope_input_positions(
            input_tokens=prompt, mm_features=features, config=self.model.hf_config
        )
        if tensor_evidence(positions) != evidence["tensors"]["position_ids"]:
            raise ContractError(
                "vLLM multimodal positions differ from processing evidence"
            )
        return mm_input(prompt, kwargs, {"image": hashes}, {"image": ranges}), evidence

    @asynccontextmanager
    async def _draw(self, name: str, request_id: str):
        async with self.lock:
            self._healthy()
            if self.mutating:
                raise ContractError("policy update is in progress")
            if self.full_weight and self._weight_publication is None:
                raise ContractError(
                    "full-weight generation requires a published policy"
                )
            if request_id in self.inflight:
                raise ContractError("duplicate in-flight request ID")
            adapter = self.adapter(name)
            self.inflight.add(request_id)
        try:
            yield adapter
        finally:
            async with self.lock:
                self.inflight.remove(request_id)

    async def _generate(
        self,
        *,
        name: str,
        request_id: str,
        prompt: list[int],
        params: dict,
        media: dict | None = None,
    ):
        async with self._draw(name, request_id) as adapter:
            try:
                sampling = self.sampling_factory(**params)
                engine_input = {"prompt_token_ids": prompt}
                evidence = None
                if media is not None:
                    engine_input, evidence = self._multimodal_input(prompt, media)
                final = await self._final_output(
                    adapter, engine_input, sampling, request_id, prompt
                )
                completion = final.outputs[0]
                tokens = list(completion.token_ids)
                if len(tokens) > params["max_tokens"]:
                    raise RuntimeError("native engine exceeded the completion budget")
                rows = completion.logprobs
                if rows is None or len(rows) != len(tokens):
                    raise RuntimeError(
                        "native engine token/logprob lengths do not match"
                    )
                clamped = []
                for token, row in zip(tokens, rows, strict=True):
                    if (
                        type(token) is not int
                        or not 0 <= token < self.output_vocab_size
                        or token not in row
                    ):
                        raise RuntimeError(
                            "native engine omitted the sampled token logprob"
                        )
                    # A greedy or near-certain draw can come back a few 1e-4
                    # above zero from the engine's log-softmax rounding; that
                    # is the bound, not an engine fault.
                    value = _bounded_logprob(row[token].logprob, token, clamped)
                    if value != row[token].logprob:
                        row[token].logprob = value
                _warn_clamped("sampled", request_id, clamped)
                if completion.finish_reason not in ("stop", "length"):
                    raise RuntimeError(
                        f"native engine did not finish normally: {completion.finish_reason!r}"
                    )
                stop = completion.stop_reason
                if stop is not None and (
                    type(stop) is not int
                    or not tokens
                    or tokens[-1] != stop
                    or completion.finish_reason != "stop"
                ):
                    raise RuntimeError(
                        "native engine returned invalid stop-token evidence"
                    )
                cached = final.num_cached_tokens
                if cached is not None and (
                    type(cached) is not int or not 0 <= cached <= len(prompt)
                ):
                    raise RuntimeError(
                        "native engine returned an invalid cached-token count"
                    )
                if not isinstance(completion.text, str):
                    raise RuntimeError("native engine omitted completion text")
                from ullm.backends.engine_validation import validate_output_text

                validate_output_text(
                    self.tokenizer,
                    prompt,
                    tokens,
                    completion.text,
                    completion.finish_reason,
                    sampling,
                )
                # Transport serialization happens only after this trainer-owned
                # wrapper has inspected the actual engine output.
                if params["logprobs"]:
                    from training.score_centering import pack_head

                    head = pack_head(
                        rows, k=params["logprobs"], vocab_size=self.output_vocab_size
                    )
                    holder = _response_head.get()
                    if holder is not None:
                        holder["head"] = head
                    # The shared serializer sees sampled-only rows, so it never
                    # expands the head into per-entry JSON dictionaries.
                    completion.logprobs = [
                        {token: row[token]}
                        for token, row in zip(tokens, rows, strict=True)
                    ]
                final.adapter = adapter.identity()
                if evidence is not None:
                    final.media_evidence = evidence
                return final
            except asyncio.CancelledError:
                # One native draw, never a retry. The iterator aborts too.
                await self.engine.abort(request_id)
                raise
            except Exception as error:
                self.poison(error)
                raise

    async def _final_output(self, adapter, engine_input, sampling, request_id, prompt):
        """One finished engine sequence for exactly this request, prompt and adapter."""
        final = None
        async for output in self.engine.generate(
            engine_input,
            sampling,
            request_id,
            lora_request=adapter.request,
        ):
            final = output
        if final is None or not final.finished or len(final.outputs) != 1:
            raise RuntimeError("native engine did not return one finished sequence")
        if (
            final.request_id != request_id
            or list(final.prompt_token_ids or []) != prompt
        ):
            raise RuntimeError("native engine changed the request ID or integer prompt")
        observed = final.lora_request
        if adapter.id == 0:
            if observed is not None:
                raise RuntimeError("base request unexpectedly used an adapter")
        elif observed is None or (
            observed.lora_int_id,
            observed.lora_name,
            observed.lora_path,
        ) != (adapter.id, adapter.name, adapter.path):
            raise RuntimeError("native engine returned a different LoRA identity")
        return final

    def get_tokenizer(self):
        """Expose the engine tokenizer for the shared engine protocol."""
        return self.tokenizer

    @property
    def model_config(self):
        """Expose the model used by the native engine."""
        return self.model

    async def abort(self, request_id: str) -> None:
        """Abort the original engine draw without resampling it."""
        await self.engine.abort(request_id)

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        *,
        lora_request=None,
        request_context=None,
    ):
        """Validate the shared engine request and yield its witnessed output."""
        self._healthy()
        if request_context:
            raise ContractError("task context must terminate at the trainer gateway")
        if lora_request is None:
            selected = self.adapter(self.base_name)
        elif isinstance(lora_request, dict):
            selected = self.adapter(lora_request.get("lora_name"))
        else:
            raise ContractError("LoRA request must be a protocol object")
        try:
            body = unpack_generation(
                {
                    "model": selected.name,
                    "request_id": request_id,
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "lora_request": lora_request,
                    "request_context": {},
                },
                model=selected.name,
                adapter=selected.identity(),
            )
            params = validate_generation(
                {key: value for key, value in body.items() if key != "media"},
                max_model_len=self.max_model_len,
                vocab_size=self.input_vocab_size,
                score_centering_top_k=self.score_centering_top_k,
            )
            unverified = set(params.get("logit_bias", {})) - self.summary_forbidden_token_ids
            if unverified:
                raise ContractError(
                    f"logit_bias names {sorted(unverified)}: only the summary-forbidden "
                    f"token ids verified at startup "
                    f"{sorted(self.summary_forbidden_token_ids)} may be biased"
                )
        except ValueError as error:
            raise ContractError(str(error)) from error
        yield await self._generate(
            name=selected.name,
            request_id=request_id,
            prompt=body["prompt"],
            params=params,
            media=body.get("media"),
        )

    @asynccontextmanager
    async def _mutation(self):
        async with self.lock:
            self._healthy()
            if self.mutating or self.inflight:
                raise ContractError("adapter controls require a drained engine")
            self.mutating = True
        try:
            yield
        except (Exception, asyncio.CancelledError) as error:
            # Failed control RPC may already have applied inside the engine.
            self.poison(error)
            raise
        finally:
            async with self.lock:
                self.mutating = False

    async def _lora_census(self, path: str) -> dict:
        """Which served LoRA layer applies each module of the adapter at ``path``.

        Read-only, before any engine mutation: an adapter with a module the
        served model would not apply, or a rank/alpha pattern vLLM would drop,
        is refused (ContractError) and the engine is left as it was
        (training/lora_census.py).
        """
        from training.lora_census import (
            CENSUS_RPC,
            lora_census,
            read_adapter,
            refuse_dropped_fields,
        )

        try:
            config, names = read_adapter(Path(path))
            refuse_dropped_fields(config)
        except ValueError as error:
            raise ContractError(f"refused LoRA adapter {path}: {error}") from error
        self._healthy()
        results = await self.engine.collective_rpc(CENSUS_RPC, args=(names,))
        served = results[0] if isinstance(results, list) and results else None
        if (
            not isinstance(served, dict)
            or set(served) != {"mapped", "modules", "packed"}
            or not isinstance(served["mapped"], list)
            or len(served["mapped"]) != len(names)
            or not isinstance(served["modules"], list)
            or not isinstance(served["packed"], dict)
            or any(result != served for result in results[1:])
        ):
            raise RuntimeError(
                f"native engine returned an invalid LoRA module census: {results!r}"[:500]
            )
        try:
            return lora_census(names, served)
        except ValueError as error:
            raise ContractError(f"refused LoRA adapter {path}: {error}") from error

    async def load_adapter(self, body: dict) -> dict:
        if self.full_weight:
            raise ContractError("full-weight engines do not load LoRA adapters")
        if not self._lora_enabled():
            raise ContractError(
                "this engine serves without LoRA; Flash applies merged policy weights"
            )
        if not isinstance(body, dict) or set(body) != {"name", "path"}:
            raise ContractError("adapter load requires only name and path")
        name, path = body["name"], body["path"]
        if not isinstance(name, str) or not name or name == self.base_name:
            raise ContractError("adapter name must be distinct from the base model")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not Path(path).is_dir()
        ):
            raise ContractError("adapter path must be an existing absolute directory")
        census = await self._lora_census(path)
        async with self._mutation():
            identifier = self.next_adapter_id
            self.next_adapter_id += 1
            request = self.lora_factory(
                name, identifier, path, base_model_name=self.base_name
            )
            old = self.adapters.get(name)
            if not await self.engine.add_lora(request):
                raise RuntimeError("native engine refused adapter load")
            if old is not None and not await self.engine.remove_lora(old.id):
                raise RuntimeError("native engine refused retiring the old adapter")
            # vLLM hashes cached blocks by LoRA name, which is reused for
            # successive policies. A fresh integer ID does not isolate K/V.
            if not await self.engine.reset_prefix_cache():
                raise RuntimeError(
                    "native engine refused prefix cache reset after adapter load"
                )
            adapter = Adapter(name, identifier, path, self.instance_id, request)
            self.adapters[name] = adapter
            log.info(
                "Loaded new LoRA adapter: name '%s', path '%s', id %d, applied %s",
                name,
                path,
                identifier,
                census["modules"],
            )
            return {"adapter": adapter.identity(), "lora_census": census}

    def _lora_enabled(self) -> bool:
        return getattr(self.engine.vllm_config, "lora_config", None) is not None

    @staticmethod
    def _worker_policy_weights(results, manifest: dict, digest: str) -> dict:
        """Every worker must report the published tensors with the learner's checksums."""
        if not isinstance(results, list) or not results:
            raise RuntimeError("native engine returned no policy weight results")
        first = results[0]
        published = dict(zip(manifest["names"], manifest["checksums"], strict=True))
        if (
            not isinstance(first, dict)
            or first.get("applied") != len(published)
            or first.get("sha256") != digest
            or first.get("iteration") != manifest["iteration"]
            or first.get("checksums") != published
        ):
            raise RuntimeError(f"native engine returned an invalid policy weight result: {first!r}"[:500])
        if any(
            {k: v for k, v in result.items() if k != "seconds"}
            != {k: v for k, v in first.items() if k != "seconds"}
            for result in results[1:]
        ):
            raise RuntimeError("native engine workers disagree on the applied policy weights")
        return first

    async def policy_weights_info(self) -> dict:
        """The published policy; nulls while the checkpoint's weights are served.

        A read: draws may be in flight, but not during a control update.
        """
        if not self.policy_weights:
            raise ContractError("native engine does not serve Flash policy weights")
        async with self.lock:
            self._healthy()
            if self.mutating:
                raise ContractError(
                    "policy weights info is unavailable during a control update"
                )
            applied = self._policy_weights or {}
        return {
            "sha256": applied.get("sha256"),
            "iteration": applied.get("iteration"),
            "tensors": applied.get("tensors", 0),
            "server_instance_id": self.instance_id,
        }

    def weight_info(self) -> dict:
        """Describe the actual NCCL participants without exposing engine controls."""
        if not (self.full_weight or self.policy_weights):
            raise ContractError("native engine is not configured for NCCL weight transfer")
        return {
            "world_size": self.engine.vllm_config.parallel_config.world_size_across_dp,
            "server_instance_id": self.instance_id,
        }

    def _weight_result(self, status: str) -> dict:
        return {"status": status, "server_instance_id": self.instance_id}

    async def init_weights(self, body: dict) -> dict:
        info = self.weight_info()
        if not isinstance(body, dict) or set(body) != {"init_info"}:
            raise ContractError("weight initialization requires init_info")
        init = body["init_info"]
        if (
            not isinstance(init, dict)
            or set(init)
            != {"master_address", "master_port", "rank_offset", "world_size"}
            or not isinstance(init["master_address"], str)
            or not init["master_address"]
            or any(
                type(init[key]) is not int
                for key in ("master_port", "rank_offset", "world_size")
            )
            or not 0 < init["master_port"] < 65536
            or init["rank_offset"] < 1
            or init["world_size"] < init["rank_offset"] + info["world_size"]
        ):
            raise ContractError("invalid native NCCL rendezvous")
        if self._weight_phase is not None:
            raise ContractError("weight transfer is already initialized")
        async with self._mutation():
            await self.engine.init_weight_transfer_engine(_weight_request("init", init))
            self._weight_phase = "idle"
        return self._weight_result("initialized")

    @asynccontextmanager
    async def _weight_operation(self, phase: str):
        """Keep all draws fenced across the complete multi-RPC publication."""
        self.weight_info()
        async with self.lock:
            self._healthy()
            if self._weight_control_active or self._weight_phase != phase:
                raise ContractError(f"weight control requires phase {phase}")
            if phase == "idle":
                if self.mutating or self.inflight:
                    raise ContractError("weight publication requires a drained engine")
                self.mutating = True
            self._weight_control_active = True
        try:
            yield
        except (Exception, asyncio.CancelledError) as error:
            # Never resume a partially loaded model or reuse its version.
            self.poison(error)
            raise
        finally:
            async with self.lock:
                self._weight_control_active = False

    @staticmethod
    def _empty_weight_body(body: dict) -> None:
        if not isinstance(body, dict) or body:
            raise ContractError("weight control requires an empty object")

    async def pause_weights(self, body: dict) -> dict:
        self._empty_weight_body(body)
        async with self._weight_operation("idle"):
            await self.engine.pause_generation(mode="wait", clear_cache=True)
            if not await self.engine.is_paused():
                raise RuntimeError("native engine did not pause for weight publication")
            self._weight_phase = "paused"
        return self._weight_result("paused")

    def _policy_publication(self, body: dict) -> tuple[dict, dict]:
        """A Flash start: the publication's label and the manifest it hashes."""
        from training.flash_policy_weights import (
            manifest_sha256,
            validate_policy_manifest,
        )
        from training.policy_sync.vllm_nccl import validate_publication

        if not isinstance(body, dict) or "manifest" not in body:
            raise ContractError("a policy weights publication starts with its manifest")
        try:
            label = validate_publication(
                {key: value for key, value in body.items() if key != "manifest"}
            )
            if label["kind"] != "policy_weights":
                raise ValueError("this engine receives Flash policy weights only")
            manifest = validate_policy_manifest(body["manifest"])
        except ValueError as error:
            raise ContractError(str(error)) from error
        if (
            manifest_sha256(manifest) != label["policy_sha256"]
            or manifest["iteration"] != label["policy_iteration"]
        ):
            raise ContractError(
                "policy weights label does not name its manifest's hash and iteration"
            )
        return label, manifest

    async def start_weights(self, body: dict) -> dict:
        from training.policy_sync.vllm_nccl import validate_publication

        if self.policy_weights:
            publication, manifest = self._policy_publication(body)
            async with self._weight_operation("paused"):
                # Each worker checks the manifest against the checkpoint and
                # places every tensor before the learner sends a byte.
                results = await self.engine.collective_rpc(
                    "prepare_flash_policy_weights", args=(manifest,)
                )
                expected = {
                    "prepared": len(manifest["names"]),
                    "sha256": publication["policy_sha256"],
                }
                if not isinstance(results, list) or not results or any(
                    result != expected for result in results
                ):
                    raise RuntimeError(
                        f"native engine did not prepare the policy weights: {results!r}"[:500]
                    )
                self._pending_weight_publication = publication
                self._pending_policy_manifest = manifest
                self._weight_phase = "started"
            return self._weight_result("started")
        try:
            publication = validate_publication(body)
            if publication["kind"] != "full_weight":
                raise ValueError("this engine receives full weights only")
        except ValueError as error:
            raise ContractError(str(error)) from error
        if publication["weight_version"] in self._weight_versions:
            raise ContractError("full-weight publication versions cannot be reused")
        async with self._weight_operation("paused"):
            self._pending_weight_publication = publication
            self._weight_versions.add(publication["weight_version"])
            await self.engine.start_weight_update()
            self._weight_phase = "started"
        return self._weight_result("started")

    async def update_weights(self, body: dict) -> dict:
        from training.policy_sync.vllm_nccl import WeightManifest

        async with self._weight_operation("started"):
            if not isinstance(body, dict) or set(body) != {"update_info"}:
                raise ContractError("weight receive requires update_info")
            try:
                manifest = WeightManifest.from_update_info(body["update_info"])
            except ValueError as error:
                raise ContractError(str(error)) from error
            if self.policy_weights:
                policy = self._pending_policy_manifest
                if (
                    body["update_info"]["packed"]
                    or list(manifest.names) != policy["names"]
                    or [list(shape) for shape in manifest.shapes] != policy["shapes"]
                ):
                    raise ContractError(
                        "weight manifest differs from the pending publication"
                    )
                # Not vLLM's update_weights: its NCCL engine reloads layerwise
                # (every layer to meta and back, partly loaded layers
                # materialized uninitialized), which a publication of some
                # tensors cannot pass through. Each worker instead takes the
                # tensors off the same NCCL group and loads them one by one.
                digest = self._pending_weight_publication["policy_sha256"]
                results = await self.engine.collective_rpc(
                    "receive_flash_policy_weights", args=(digest,)
                )
                if not isinstance(results, list) or not results or any(
                    not isinstance(result, dict)
                    or result.get("received") != len(policy["names"])
                    or result.get("sha256") != digest
                    for result in results
                ):
                    raise RuntimeError(
                        f"native engine did not receive the policy weights: {results!r}"[:500]
                    )
                log.info(
                    "Received and loaded %d policy weights (sha256 %s) in %s s per worker",
                    len(policy["names"]),
                    digest,
                    "/".join(str(result.get("seconds")) for result in results),
                )
                self._weight_phase = "received"
                return self._weight_result("received")
            if manifest.sha256 != self._pending_weight_publication["manifest_sha256"]:
                raise ContractError(
                    "weight manifest differs from the pending publication"
                )
            await self.engine.update_weights(
                _weight_request("update", body["update_info"])
            )
            self._weight_phase = "received"
        return self._weight_result("received")

    async def finish_weights(self, body: dict) -> dict:
        self._empty_weight_body(body)
        async with self._weight_operation("received"):
            if self.policy_weights:
                started = time.perf_counter()
                manifest = self._pending_policy_manifest
                digest = self._pending_weight_publication["policy_sha256"]
                # Each worker recomputes every checksum from the served rows.
                applied = self._worker_policy_weights(
                    await self.engine.collective_rpc(
                        "verify_flash_policy_weights", args=(digest,)
                    ),
                    manifest,
                    digest,
                )
                log.info(
                    "Verified %d policy weights of iteration %d (sha256 %s) in %.2fs",
                    applied["applied"],
                    applied["iteration"],
                    digest,
                    time.perf_counter() - started,
                )
            else:
                await self.engine.finish_weight_update()
            # Cached K/V and recurrent states were computed with the old weights.
            if not await self.engine.reset_prefix_cache():
                raise RuntimeError(
                    "native engine refused cache reset after weight publication"
                )
            self._weight_phase = "finished"
        return self._weight_result("finished")

    async def resume_weights(self, body: dict) -> dict:
        self._empty_weight_body(body)
        async with self._weight_operation("finished"):
            await self.engine.resume_generation()
            self._weight_publication = self._pending_weight_publication
            if self.policy_weights:
                manifest = self._pending_policy_manifest
                self._policy_weights = {
                    "sha256": self._weight_publication["policy_sha256"],
                    "iteration": manifest["iteration"],
                    "tensors": len(manifest["names"]),
                }
                self._pending_policy_manifest = None
            self._pending_weight_publication = None
            self._weight_phase = "idle"
            self.mutating = False
        return self._weight_result("resumed") | {
            "adapter": self.adapter(self.base_name).identity()
        }

    async def unload_adapter(self, body: dict) -> dict:
        if (
            not isinstance(body, dict)
            or set(body) != {"name"}
            or not isinstance(body["name"], str)
        ):
            raise ContractError("adapter unload requires only name")
        async with self._mutation():
            old = self.adapters.get(body["name"])
            if old is not None:
                if not await self.engine.remove_lora(old.id):
                    raise RuntimeError("native engine refused adapter unload")
                if not await self.engine.reset_prefix_cache():
                    raise RuntimeError(
                        "native engine refused prefix cache reset after adapter unload"
                    )
                del self.adapters[old.name]
            return {"removed": old is not None}

    async def warm_adapter(self, body: dict):
        if (
            not isinstance(body, dict)
            or set(body) != {"name"}
            or not isinstance(body["name"], str)
            or not body["name"]
        ):
            raise ContractError("adapter warm-up requires only a nonempty name")
        prompt = self.tokenizer.encode("0", add_special_tokens=False)
        return await self._generate(
            name=body["name"],
            request_id=f"trainer-warm#{uuid.uuid4().hex}",
            prompt=prompt,
            params={
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "min_p": 0.0,
                "min_tokens": 0,
                "logprobs": 0,
                "skip_special_tokens": False,
                "ignore_eos": False,
                "detokenize": True,
            },
        )


def create_app(
    service: NativeService, *, api_key: str | None = None
) -> web.Application:
    @web.middleware
    async def boundary(request, handler):
        if api_key and not hmac.compare_digest(
            request.headers.get("Authorization", ""), f"Bearer {api_key}"
        ):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            service._healthy()
            marker = _response_head.set({})
            try:
                response = await handler(request)
                head = _response_head.get().get("head")
                if head is not None:
                    payload = json.loads(response.body)
                    payload["sampler_topk"] = head
                    response = web.json_response(payload)
                return response
            finally:
                _response_head.reset(marker)
        except (ContractError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        except web.HTTPException:
            raise
        except Exception as error:
            service.poison(error)
            return web.json_response({"error": str(error)}, status=503)

    from ullm.backends.engine_proxy import create_engine_app

    # The reusable transport has no training rules. This repository supplies
    # the checked engine wrapper and owns its lifecycle and control routes.
    app = create_engine_app(
        service,
        service.metadata,
        sampling_factory=lambda **kwargs: kwargs,
        lora_factory=lambda **kwargs: kwargs,
    )
    app.middlewares.insert(0, boundary)

    async def health(request):
        await service.engine.check_health()
        return web.json_response({"status": "ok", "transport": TRANSPORT})

    async def runtime(request):
        result = runtime_config(service.engine)
        result["score_centering_top_k"] = service.score_centering_top_k
        _validate_served_runtime(result, service.score_centering_top_k)
        return web.json_response(result)

    async def models(request):
        return web.json_response(
            {
                "models": [service.adapter(service.base_name).identity()]
                + [item.identity() for item in service.adapters.values()],
                "max_model_len": service.max_model_len,
                "server_instance_id": service.instance_id,
            }
        )

    async def metrics(request):
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return web.Response(
            body=generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    async def weights_info(request):
        return web.json_response(service.weight_info())

    async def policy_weights_info(request):
        return web.json_response(await service.policy_weights_info())

    for path, handler in (
        ("/health", health),
        ("/runtime", runtime),
        ("/models", models),
        ("/metrics", metrics),
        ("/weights/info", weights_info),
        ("/policy_weights/info", policy_weights_info),
    ):
        app.router.add_get(path, handler)
    for operation in ("load", "unload", "warm"):
        method = getattr(service, f"{operation}_adapter")

        async def control(request, method=method, operation_name=operation):
            result = await method(await request.json())
            if operation_name == "warm":
                from ullm.backends.engine_proxy import serialize_output

                result = serialize_output(result)
            return web.json_response(result)

        app.router.add_post(f"/adapters/{operation}", control)
    for operation in ("init", "pause", "start", "update", "finish", "resume"):
        method = getattr(service, f"{operation}_weights")

        async def weights_control(request, method=method):
            return web.json_response(await method(await request.json()))

        app.router.add_post(f"/weights/{operation}", weights_control)
    return app


def configure_engine_worker(engine_args) -> None:
    """Carry the mandatory guard into spawned vLLM workers, not just the parent.

    LoRA engines also get the census extension, which the gateway asks which
    served layer applies each adapter module before it loads an adapter.
    """
    from training.lora_census import WORKER_EXTENSION

    worker = "training.vllm_blackwell_worker.BlackwellGdnWorker"
    if engine_args.worker_cls not in ("auto", worker):
        raise RuntimeError("native serving requires the guarded vLLM worker")
    engine_args.worker_cls = worker
    if getattr(engine_args, "enable_lora", False):
        if engine_args.worker_extension_cls not in ("", WORKER_EXTENSION):
            raise RuntimeError("native LoRA serving requires the LoRA census worker extension")
        engine_args.worker_extension_cls = WORKER_EXTENSION


def main() -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--score-centering-top-k", type=int, default=0)
    parser.add_argument(
        "--summary-forbidden-token-ids", type=summary_forbidden_token_ids, default=()
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    validate_bind_auth(args.host, os.environ.get("SERVER_API_KEY"))
    version = importlib.metadata.version("vllm")
    if version != "0.25.1" and not is_flash_vllm_version(version):
        raise RuntimeError("native serving requires vLLM 0.25.1 or Flash fork 3bda1ba09")

    engine_args = AsyncEngineArgs.from_cli_args(args)
    configure_engine_worker(engine_args)
    if args.score_centering_top_k < 0:
        raise ValueError("score-centering-top-k must be nonnegative")
    engine_args.max_logprobs = max(engine_args.max_logprobs, args.score_centering_top_k)

    async def application():
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        try:
            service = NativeService(
                engine,
                score_centering_top_k=args.score_centering_top_k,
                summary_forbidden_token_ids=args.summary_forbidden_token_ids,
                # On the Flash fork the NCCL routes carry merged policy weights.
                policy_weights=is_flash_vllm_version(version)
                and getattr(engine_args, "weight_transfer_config", None) is not None,
            )
            app = create_app(service, api_key=os.environ.get("SERVER_API_KEY"))
        except BaseException:
            engine.shutdown()
            raise

        async def statistics():
            try:
                while True:
                    await asyncio.sleep(10)
                    await engine.do_log_stats()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                service.poison(error)

        stats_task = asyncio.create_task(statistics())

        async def cleanup(app):
            stats_task.cancel()
            await asyncio.gather(stats_task, return_exceptions=True)
            engine.shutdown()

        app.on_cleanup.append(cleanup)
        return app

    logging.basicConfig(level=logging.INFO)
    # Continue to a definite outcome after TCP disconnect. Gateway drains its
    # own request; no manual generation timeout or background retry is allowed.
    web.run_app(
        application(),
        host=args.host,
        port=args.port,
        handler_cancellation=False,
        keepalive_timeout=300,
        shutdown_timeout=60,
    )


if __name__ == "__main__":
    main()
