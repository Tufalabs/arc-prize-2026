"""Single-attempt engine transport with local rendering and owned request drains.

The wire carries engine inputs and outputs plus an opaque request context.
It contains no admission, reward, history or compaction policy. GPU and
tokenizer dependencies are imported only when connecting or serving an engine.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import types
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal

import httpx
import ullm.backend
import ullm.backends.engine_interface
import ullm.factory
import ullm.support

TRANSPORT = "engine-proxy-v1"
DEFAULT_GENERATION_TIMEOUT_S = 900.0


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, str]:
    """Hash the full tokenizer encoding rules, independently of its path."""
    serialized = json.loads(tokenizer.backend_tokenizer.to_str())
    canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"algorithm": "hf-tokenizers-json-sha256-v1", "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def sampling_to_wire(params: Any) -> dict[str, Any]:
    """Preserve every supplied sampling field, refusing unsupported objects."""
    if isinstance(params, dict):
        return dict(params)
    import msgspec

    baseline = type(params)()
    required = {
        "n",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "min_tokens",
        "ignore_eos",
        "skip_special_tokens",
        "max_tokens",
        "logprobs",
    }
    optional = {
        "seed",
        "stop",
        "stop_token_ids",
        "prompt_logprobs",
        "logprob_token_ids",
        "flat_logprobs",
        "detokenize",
        "spaces_between_special_tokens",
        "include_stop_str_in_output",
        "output_kind",
        "stream_interval",
        "skip_clone",
        "structured_outputs",
        "logit_bias",
        "allowed_token_ids",
        "extra_args",
        "routed_experts_prompt_start",
        "bad_words",
        "skip_reading_prefix_cache",
        "thinking_token_budget",
        "repetition_detection",
        "zero_recurrent_state",
        "zero_recurrent_at",
    }
    fields = msgspec.structs.fields(type(params))
    public = {
        info.name: getattr(params, info.name)
        for info in fields
        if info.name in required
        or (info.name in optional and getattr(params, info.name) != getattr(baseline, info.name))
    }
    # Constructor-derived private fields may be omitted only when rebuilding
    # the public inputs reproduces them. Unknown nonneutral fields fail closed.
    reconstructed = type(params)(**public)
    for info in fields:
        if info.name not in public and getattr(params, info.name) != getattr(reconstructed, info.name):
            raise ValueError(f"engine proxy cannot preserve sampling field {info.name}")
    return msgspec.to_builtins(public)


def lora_to_wire(request: Any) -> dict[str, Any] | None:
    """Copy the engine's immutable adapter coordinates without coercion."""
    if request is None:
        return None
    names = ("lora_name", "lora_int_id", "lora_path")
    result = {name: request[name] if isinstance(request, dict) else getattr(request, name) for name in names}
    if (
        not isinstance(result["lora_name"], str)
        or type(result["lora_int_id"]) is not int
        or result["lora_int_id"] <= 0
        or not isinstance(result["lora_path"], str)
    ):
        raise ValueError("invalid engine LoRA identity")
    return result


def _token_ids(value: Any) -> list[int]:
    """Refuse lossy token conversions at either end of the wire."""
    if not isinstance(value, (list, tuple)) or any(type(t) is not int or t < 0 for t in value):
        raise ValueError("engine token IDs must be exact nonnegative integers")
    return list(value)


def output_to_wire(output: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Serialize an actual engine output without re-rendering or decoding it."""
    completions = []
    for completion in output.outputs:
        tokens = _token_ids(completion.token_ids)
        rows = completion.logprobs
        if rows is not None:
            if len(rows) != len(tokens):
                raise ValueError("engine token/logprob lengths differ")
            rows = [
                {
                    str(token): {
                        "logprob": value.logprob,
                        "rank": value.rank,
                        "decoded_token": getattr(value, "decoded_token", None),
                    }
                    for token, value in row.items()
                }
                for row in rows
            ]
        completions.append(
            {
                "index": completion.index,
                "text": completion.text,
                "token_ids": tokens,
                "logprobs": rows,
                "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
        )
    lora = lora_to_wire(getattr(output, "lora_request", None))
    adapter = getattr(output, "adapter", None)
    if adapter is None:
        declared = metadata.get("adapter") or {}
        adapter = ullm.backends.engine_interface.adapter_identity(
            lora,
            model=declared.get("name", metadata.get("model_id")),
            server_instance_id=declared.get("server_instance_id"),
        )
    return {
        "request_id": output.request_id,
        "prompt_token_ids": _token_ids(output.prompt_token_ids),
        "outputs": completions,
        "finished": output.finished,
        "num_cached_tokens": output.num_cached_tokens,
        "lora_request": lora,
        "adapter": adapter,
        **({"media_evidence": output.media_evidence} if getattr(output, "media_evidence", None) is not None else {}),
    }


def serialize_output(output: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Expose the common output envelope to engine owners' utility endpoints."""
    return output_to_wire(output, metadata or {})


def output_from_wire(payload: Any) -> Any:
    """Restore vLLM-shaped evidence, retaining the exact engine text."""
    if not isinstance(payload, dict) or payload.get("finished") is not True:
        raise ValueError("engine proxy requires a finished engine output")
    prompt = _token_ids(payload.get("prompt_token_ids"))
    cached = payload.get("num_cached_tokens")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= len(prompt)):
        raise ValueError("invalid engine cached-token count")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("engine proxy returned no completions")
    completions = []
    for completion in outputs:
        tokens = _token_ids(completion.get("token_ids"))
        if not isinstance(completion.get("text"), str):
            raise ValueError("engine completion text must be a string")
        if completion.get("finish_reason") not in ("stop", "length"):
            raise ValueError("engine generation did not finish normally")
        rows = completion.get("logprobs")
        restored = None
        if rows is not None:
            if not isinstance(rows, list) or len(rows) != len(tokens):
                raise ValueError("engine token/logprob lengths differ")
            restored = []
            for token, row in zip(tokens, rows, strict=True):
                if not isinstance(row, dict) or str(token) not in row:
                    raise ValueError("engine omitted a sampled token logprob")
                values = {}
                for key, value in row.items():
                    if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
                        raise ValueError("engine logprob token ID is not canonical")
                    probability = value.get("logprob")
                    if type(probability) not in (int, float) or not math.isfinite(probability) or probability > 0:
                        raise ValueError("engine logprobs must be finite and nonpositive")
                    values[int(key)] = types.SimpleNamespace(**value)
                restored.append(values)
        completions.append(types.SimpleNamespace(**{**completion, "token_ids": tokens, "logprobs": restored}))
    lora = payload.get("lora_request")
    lora = None if lora is None else types.SimpleNamespace(**lora_to_wire(lora))
    return types.SimpleNamespace(
        **{**payload, "prompt_token_ids": prompt, "outputs": completions, "lora_request": lora}
    )


def _consume_finished(future: asyncio.Future[Any]) -> None:
    """Retrieve a timed-out task's eventual error without delaying teardown."""
    if not future.cancelled():
        future.exception()


async def _drain(
    future: asyncio.Future[Any], *, timeout_s: float = DEFAULT_GENERATION_TIMEOUT_S
) -> tuple[Any, asyncio.CancelledError | None]:
    """Preserve owned draws through cancellation, but never beyond the deadline."""
    cancelled = None
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not future.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            future.cancel()
            future.add_done_callback(_consume_finished)
            raise TimeoutError("engine generation drain exceeded its deadline; outcome is uncertain")
        try:
            await asyncio.wait({future}, timeout=remaining)
        except asyncio.CancelledError as error:
            cancelled = cancelled or error
    return future.result(), cancelled


@dataclass
class EngineModelConfig(ullm.support.BaseClass):
    """The model metadata read by local vllmx rendering and trace code."""

    model: str
    revision: str | None
    tokenizer: str
    tokenizer_revision: str | None
    max_model_len: int | None
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_multimodal_config(self) -> Any:
        """Return the processor settings declared by the serving engine."""
        return types.SimpleNamespace(mm_processor_kwargs=self.mm_processor_kwargs)


@dataclass
class EngineProxy(ullm.support.BaseClass):
    """Picklable, single-attempt engine transport with whole-run fatal errors.

    Any failed or uncertain draw, including a transport error or deadline,
    permanently poisons this proxy and is raised again at close. Hosts must
    abort the run even if a game's error handler swallows the original error;
    reconnecting or replaying would lose ownership of the admitted generation.
    Tokenizer code from a server-declared path is never trusted. Callers that
    need custom tokenizer code must select a path and explicitly opt in.
    """

    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    tokenizer_path: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    generation_timeout_s: float = DEFAULT_GENERATION_TIMEOUT_S
    remote_engine: Literal[True] = field(default=True, init=False)
    unix_socket: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, init=False)
    model_config: Any = field(default=None, init=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _http: Any = field(default=None, init=False, repr=False)
    _connect_lock: Any = field(default=None, init=False, repr=False)
    _requests: dict[str, asyncio.Task[Any]] = field(default_factory=dict, init=False, repr=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        if self._requests:
            raise TypeError("cannot pickle an engine proxy with owned requests")
        return {
            name: getattr(self, name)
            for name in (
                "base_url",
                "model",
                "api_key",
                "tokenizer_path",
                "tokenizer_revision",
                "local_files_only",
                "trust_remote_code",
                "generation_timeout_s",
                "unix_socket",
            )
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(type(self)(**state).__dict__)

    def _client(self) -> httpx.AsyncClient:
        """Create one nonretrying transport; owned drains bound total draw time."""
        if self._closing:
            raise RuntimeError("engine proxy is closing")
        if self._http is None:
            self._http = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(
                    uds=self.unix_socket,
                    retries=0,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=32),
                ),
                timeout=httpx.Timeout(self.generation_timeout_s, connect=30),
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            )
        return self._http

    def _load_tokenizer(self, metadata: dict[str, Any]) -> Any:
        """Load the declared local tokenizer and compare all rendering identity."""
        import transformers

        if self.trust_remote_code and self.tokenizer_path is None:
            raise ValueError("custom tokenizer code requires an explicit tokenizer_path")
        revision = metadata.get("tokenizer_revision")
        if self.tokenizer_revision is not None and self.tokenizer_revision != revision:
            raise ValueError("configured tokenizer revision differs from engine metadata")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_path or metadata["tokenizer_id"],
            revision=revision or None,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        for name in ("chat_template", "special_tokens_map", "bos_token_id", "eos_token_id", "pad_token_id"):
            if name not in metadata or getattr(tokenizer, name) != metadata[name]:
                raise ValueError(f"local tokenizer {name} differs from engine metadata")
        if tokenizer_fingerprint(tokenizer) != metadata.get("tokenizer_fingerprint"):
            raise ValueError("local tokenizer fingerprint differs from engine metadata")
        return tokenizer

    async def connect(self) -> None:
        """Verify rendering identity before any local prompt is prepared."""
        if self._failure is not None:
            raise RuntimeError("engine proxy has failed") from self._failure
        if not math.isfinite(self.generation_timeout_s) or self.generation_timeout_s <= 0:
            raise ValueError("engine generation timeout must be finite and positive")
        if self.metadata:
            return
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self.metadata:
                return
            response = await self._client().get(
                f"{self.base_url.rstrip('/')}/metadata", params={"model": self.model}, timeout=30
            )
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("transport") != TRANSPORT:
                raise ValueError("server does not advertise the engine proxy protocol")
            server_deadline = metadata.get("max_generation_seconds", self.generation_timeout_s)
            if type(server_deadline) not in (int, float) or not math.isfinite(server_deadline) or server_deadline <= 0:
                raise ValueError("engine generation deadline must be finite and positive")
            self.generation_timeout_s = min(self.generation_timeout_s, server_deadline)
            tokenizer = await asyncio.to_thread(self._load_tokenizer, metadata)
            self.model_config = EngineModelConfig(
                model=metadata["model_id"],
                revision=metadata.get("model_revision"),
                tokenizer=metadata["tokenizer_id"],
                tokenizer_revision=metadata.get("tokenizer_revision"),
                max_model_len=metadata.get("max_model_len"),
                mm_processor_kwargs=metadata.get("mm_processor_kwargs", {}),
            )
            self._tokenizer = tokenizer
            self.metadata = metadata

    def get_tokenizer(self) -> Any:
        """Return the verified local tokenizer; connecting is explicit and async."""
        if self._tokenizer is None:
            raise RuntimeError("await EngineProxy.connect() before rendering")
        return self._tokenizer

    def get_processor(self) -> Any:
        """Load and verify the exact processor declared by the native engine."""
        if self._processor is None:
            import transformers
            import ullm.media

            identity = self.metadata.get("processing")
            if identity is None:
                raise ValueError("engine does not advertise verified image processing")
            processor = transformers.AutoProcessor.from_pretrained(
                self.tokenizer_path or identity["model"],
                revision=identity["revision"],
                local_files_only=self.local_files_only,
                trust_remote_code=True,
                **identity["mm_processor_kwargs"],
            )
            actual = ullm.media.processing_identity(
                processor,
                model=identity["model"],
                revision=identity["revision"],
                mm_processor_kwargs=identity["mm_processor_kwargs"],
                model_architecture=identity["model_architecture"],
            )
            if actual != identity:
                raise ValueError("local image processing differs from engine metadata")
            self._processor = processor
        return self._processor

    def prepare_media(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Serialize all images with their order, expansion and processing proof."""
        import ullm.media

        supplied = prompt.get("multi_modal_data")
        if not isinstance(supplied, dict) or set(supplied) != {"image"}:
            raise ValueError("native image transport requires only ordered images")
        identity = self.metadata.get("processing")
        if identity is None:
            raise ValueError("engine does not advertise verified image processing")
        effective = {**identity["mm_processor_kwargs"], **prompt.get("mm_processor_kwargs", {})}
        if effective != identity["mm_processor_kwargs"]:
            raise ValueError("image processor overrides differ from engine metadata")
        return ullm.media.build_media(
            _token_ids(prompt["prompt_token_ids"]), supplied["image"], self.get_processor(), identity
        )

    @property
    def lora_request(self) -> Any:
        """Resolve the connected alias to its verified immutable LoRA identity."""
        adapter = self.metadata.get("adapter")
        if not adapter or adapter["id"] == 0:
            return None
        return types.SimpleNamespace(lora_name=adapter["name"], lora_int_id=adapter["id"], lora_path=adapter["path"])

    async def allows_generation_replay(self) -> bool:
        """An uncertain native draw never authorizes another attempt."""
        return False

    async def _request(self, body: dict[str, Any]) -> Any:
        """Read and validate exactly one HTTP response, recording transport failure."""
        try:
            response = await self._http.post(f"{self.base_url.rstrip('/')}/generate", json=body)
            response.raise_for_status()
            output = output_from_wire(response.json())
            if output.request_id != body["request_id"] or output.prompt_token_ids != body["prompt"]["prompt_token_ids"]:
                raise ValueError("engine changed the request ID or integer prompt")
            if lora_to_wire(output.lora_request) != body["lora_request"]:
                raise ValueError("engine returned a different LoRA identity")
            expected = self.metadata.get("adapter")
            if expected is not None and output.adapter != expected:
                raise ValueError("engine adapter or server identity changed after connection")
            media = body["prompt"].get("media")
            if media is not None:
                expected_evidence = {key: value for key, value in media.items() if key not in {"version", "images"}}
                if getattr(output, "media_evidence", None) != expected_evidence:
                    raise ValueError("engine consumed different image processing evidence")
            elif getattr(output, "media_evidence", None) is not None:
                raise ValueError("text-only engine output unexpectedly carries image evidence")
            return output
        except Exception as error:
            self._failure = error
            raise

    async def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: Any = None,
        request_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        """Yield one complete draw, draining it before propagating cancellation."""
        await self.connect()
        if self._closing:
            raise RuntimeError("engine proxy is closing")
        if not isinstance(request_id, str) or not request_id or request_id in self._requests:
            raise ValueError("engine generation requires a unique nonempty request ID")
        if not isinstance(prompt, dict) or set(prompt) - {
            "type",
            "prompt_token_ids",
            "multi_modal_data",
            "mm_processor_kwargs",
            "media",
        }:
            raise ValueError("engine proxy currently supports exact token prompts")
        body: dict[str, Any] = {
            "model": self.model,
            "request_id": request_id,
            "prompt": {"type": "token", "prompt_token_ids": _token_ids(prompt.get("prompt_token_ids"))},
            "sampling_params": sampling_to_wire(sampling_params),
            "lora_request": lora_to_wire(lora_request),
            "request_context": dict(request_context or {}),
        }
        if "multi_modal_data" in prompt:
            body["prompt"]["media"] = self.prepare_media(prompt)
        elif "media" in prompt:
            import ullm.media

            ullm.media.validate_media(prompt["media"], body["prompt"]["prompt_token_ids"])
            body["prompt"]["media"] = prompt["media"]
        task = asyncio.create_task(self._request(body))
        self._requests[request_id] = task
        try:
            output, cancelled = await _drain(task, timeout_s=self.generation_timeout_s)
            if cancelled is not None:
                raise cancelled
            yield output
        except Exception as error:
            self._failure = error
            raise
        finally:
            self._requests.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        """Drain an admitted draw until completion or a fatal uncertain timeout."""
        task = self._requests.get(request_id)
        if task is not None:
            _, cancelled = await _drain(task, timeout_s=self.generation_timeout_s)
            if cancelled is not None:
                raise cancelled

    async def aclose(self) -> None:
        """Drain every owned response before closing the shared HTTP connection."""
        self._closing = True
        cancelled = None
        try:
            if self._requests:
                _, cancelled = await _drain(
                    asyncio.gather(*self._requests.values(), return_exceptions=True),
                    timeout_s=self.generation_timeout_s,
                )
        finally:
            if self._http is not None:
                await self._http.aclose()
                self._http = None
        if self._failure is not None:
            raise self._failure
        if cancelled is not None:
            raise cancelled


@dataclass
class EngineProxyFactory(ullm.factory.BackendFactory):
    """Construct a direct vLLM backend around a caller-owned engine transport.

    Rendering and token continuation live in the host's VLLM backend; only
    engine inputs cross the transport. This never constructs a chat client.
    The factory owns the backend and drains the proxy when it closes.
    """

    engine: EngineProxy = field(kw_only=True)
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    _backend: Any = field(default=None, init=False, repr=False)

    async def _start(self) -> ullm.backend.Backend:
        """Build the direct backend in the host, leaving game children CPU-only."""
        import ullm.backends.vllmx

        if not isinstance(self.engine, EngineProxy):
            raise TypeError("EngineProxyFactory requires an EngineProxy, not a chat backend")
        self._backend = ullm.backends.vllmx.VLLM(engine=self.engine, **self.backend_kwargs)
        return self._backend

    async def _close(self) -> None:
        """Drain the direct backend's requests and release its local resources."""
        await self._backend.close()


def _sampling_factory(**fields: Any) -> Any:
    """Construct real engine sampling only in the serving process."""
    import vllm

    return vllm.SamplingParams(**fields)


def _lora_factory(**fields: Any) -> Any:
    """Construct a real adapter request only in the serving process."""
    import vllm.lora.request

    return vllm.lora.request.LoRARequest(**fields)


def create_engine_app(
    engine: Any,
    metadata: dict[str, Any] | Callable[[str | None], dict[str, Any]],
    *,
    sampling_factory: Callable[..., Any] | None = None,
    lora_factory: Callable[..., Any] | None = None,
    max_generation_seconds: float = DEFAULT_GENERATION_TIMEOUT_S,
    max_seen_requests: int = 1_000_000,
) -> Any:
    """Serve bounded single-attempt draws; owners may add lifecycle routes.

    Fixed-size ID digests stay reserved for this server's lifetime. At the
    admission cap the server refuses further draws instead of evicting IDs
    and allowing replay; the owner must retire this instance for more work.
    """
    import aiohttp.web

    if not math.isfinite(max_generation_seconds) or max_generation_seconds <= 0:
        raise ValueError("engine generation deadline must be finite and positive")
    if type(max_seen_requests) is not int or max_seen_requests <= 0:
        raise ValueError("engine request admission cap must be a positive integer")
    pending: set[asyncio.Task[Any]] = set()
    seen: set[bytes] = set()
    closing = False

    def describe(model: str | None) -> dict[str, Any]:
        return {
            **(metadata(model) if callable(metadata) else metadata),
            "transport": TRANSPORT,
            "max_generation_seconds": max_generation_seconds,
        }

    def encode_json(value: Any) -> str:
        return json.dumps(value, allow_nan=False)

    async def get_metadata(request: Any) -> Any:
        return aiohttp.web.json_response(describe(request.query.get("model")))

    async def generate(request: Any) -> Any:
        body = await request.json()
        if closing:
            raise aiohttp.web.HTTPServiceUnavailable(text="engine server is closing")
        if not isinstance(body, dict) or set(body) != {
            "model",
            "request_id",
            "prompt",
            "sampling_params",
            "lora_request",
            "request_context",
        }:
            raise aiohttp.web.HTTPBadRequest(text="invalid engine request envelope")
        if not isinstance(body["model"], str) or not body["model"]:
            raise aiohttp.web.HTTPBadRequest(text="invalid engine model alias")
        # Resolve the alias before admission and keep its identity for this draw.
        request_metadata = copy.deepcopy(describe(body["model"]))
        request_id = body["request_id"]
        if not isinstance(request_id, str) or not request_id or len(request_id) > 512:
            raise aiohttp.web.HTTPBadRequest(text="duplicate or invalid engine request ID")
        request_digest = hashlib.sha256(request_id.encode()).digest()
        if request_digest in seen:
            raise aiohttp.web.HTTPBadRequest(text="duplicate or invalid engine request ID")
        if len(seen) >= max_seen_requests:
            raise aiohttp.web.HTTPServiceUnavailable(text="engine request admission cap reached; retire this instance")
        prompt = body["prompt"]
        if (
            not isinstance(prompt, dict)
            or set(prompt) - {"type", "prompt_token_ids", "media"}
            or prompt.get("type") != "token"
        ):
            raise aiohttp.web.HTTPBadRequest(text="engine proxy requires a token prompt")
        _token_ids(prompt["prompt_token_ids"])
        if "media" in prompt:
            import ullm.media

            ullm.media.validate_media(prompt["media"], prompt["prompt_token_ids"])
        context = body["request_context"]
        if not isinstance(context, dict) or not isinstance(body["sampling_params"], dict):
            raise aiohttp.web.HTTPBadRequest(text="invalid engine request context or sampling")
        params = (sampling_factory or _sampling_factory)(**body["sampling_params"])
        lora = lora_to_wire(body["lora_request"])
        kwargs: dict[str, Any] = {}
        if lora is not None:
            kwargs["lora_request"] = (lora_factory or _lora_factory)(**lora)
        if context:
            kwargs["request_context"] = context
        seen.add(request_digest)

        async def collect() -> dict[str, Any]:
            final = None
            async for output in engine.generate(prompt, params, request_id, **kwargs):
                final = output
            if final is None or not final.finished:
                raise RuntimeError("engine did not return a completed draw")
            return output_to_wire(final, request_metadata)

        task = asyncio.create_task(collect())
        pending.add(task)
        try:
            output, cancelled = await _drain(task, timeout_s=max_generation_seconds)
            if cancelled is not None:
                raise cancelled
            return aiohttp.web.json_response(output, dumps=encode_json)
        finally:
            pending.discard(task)

    async def shutdown(app: Any) -> None:
        nonlocal closing
        closing = True
        if pending:
            results, cancelled = await _drain(
                asyncio.gather(*pending, return_exceptions=True), timeout_s=max_generation_seconds
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if cancelled is not None:
                raise cancelled

    app = aiohttp.web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_get("/metadata", get_metadata)
    app.router.add_post("/generate", generate)
    app.on_shutdown.append(shutdown)
    return app


@dataclass
class EngineServerHandle(ullm.support.BaseClass):
    """Own the HTTP listener without taking ownership of the served engine."""

    runner: Any
    base_url: str
    unix_path: str | None = None

    async def aclose(self) -> None:
        """Stop admission and drain accepted engine requests before cleanup."""
        await self.runner.cleanup()


async def start_engine_server(
    engine: Any,
    metadata: dict[str, Any] | Callable[[str | None], dict[str, Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    unix_path: str | None = None,
    sampling_factory: Callable[..., Any] | None = None,
    lora_factory: Callable[..., Any] | None = None,
    max_generation_seconds: float = DEFAULT_GENERATION_TIMEOUT_S,
    max_seen_requests: int = 1_000_000,
) -> EngineServerHandle:
    """Bind the same engine protocol on either TCP or a Unix-domain socket."""
    import aiohttp.web

    app = create_engine_app(
        engine,
        metadata,
        sampling_factory=sampling_factory,
        lora_factory=lora_factory,
        max_generation_seconds=max_generation_seconds,
        max_seen_requests=max_seen_requests,
    )
    runner = aiohttp.web.AppRunner(app, access_log=None)
    await runner.setup()
    site = aiohttp.web.UnixSite(runner, unix_path) if unix_path else aiohttp.web.TCPSite(runner, host, port)
    await site.start()
    if unix_path:
        return EngineServerHandle(runner, "http://engine", unix_path)
    address: Any = runner.addresses[0]
    return EngineServerHandle(runner, f"http://{host}:{address[1]}")
