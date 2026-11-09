# ullm design

*Register: as-built — updates with the code. Detail beyond what is here
lives in module and class docstrings; this is the map, not the manual.*

ullm is a backend-agnostic LLM conversation framework. The core idea:
conversations exist in a **user format** (backend-agnostic, what agents
build and read) and a **model format** (backend-specific, what the LLM
sees); backends own the conversion in both directions.

## Foundations

- **support.py** — `BaseClass` (frozen-after-`__init__` dataclass with
  runtime type/constraint checking; see AGENTS.md) and `Processor`, an
  async state machine (`ready`/`running`/`completed`/`cancelled`/
  `errored`) with peek/rearm; `ProcessorMulti` fans out child
  processors. `Tool`, `ResponseManager` and `TurnManager` are
  Processors; how much of that machinery survives the turn-runner
  inversion is an open roadmap item. `diagnose_pickle(obj)` returns
  dotted paths to whatever blocks pickling.
- **conv.py** — the user-format conversation model: content types
  (`StringContent`, `ListContent`, `EmptyContent`, `ToolCallContent`,
  `ImageContent` — numpy `(H, W, 3)` uint8 RGB) and elements
  (`SystemPrompt` with `tools` + `reasoning_effort` (default `"high"`),
  `UserPrompt`, `LLMResponse`, `ToolResult`, `ResponseTurn`,
  `Conversation` with a dedicated `system_prompt` field). `LLMResponse`
  splits `reasoning_content` from `final_content` and carries
  `token_usage`, a normalized `stop_reason`, three manager-written
  flags every backend honors: `do_not_send_to_agent` (synthetic
  placeholders are never sent to the model), `malformed_retries`,
  `truncated_by_per_turn_cap`, and one backend-written marker,
  `malformed_error`: why a reply the backend could not parse was kept
  as the model's turn, with content empty or raw text retained
  (`Backend.accept_malformed_output`). This marker also survives a
  `max_tokens` stop; otherwise the normalized stop is `malformed`.
  Elements cache raw backend output in
  `_model_contents` (reasoning round-tripping reads it back);
  `strip_model_contents` clears it on copies destined for logs.

## The backend seam (backend.py)

- The primary public operation is
  ``async respond(sampling_params, conversation, run_id=None)`` — an
  async generator that appends the output element, streams parsed
  user-format partials, and leaves the processed element on the
  conversation. Subclasses implement private hooks
  (`_prepare_input` / `_generate` / `_process_output` /
  `_process_partial_output`) that the default `respond()` composes; the
  hooks are extension points, not API. `run_id` attributes the call to
  a game for tracing; a backend with a `trace_recorder` refuses
  unattributed calls.
- **Response injection** (`GenericSamplingParameters.response_injection`):
  caller text served as ids of its own after the rendered generation
  prompt (`encode(text, add_special_tokens=False)`), so the model continues
  from it — inside the opened think block on a thinking template. The
  answer's text starts with it everywhere (the parsed response, history
  re-renders, the diagnostics log; `LLMResponse.response_injection` names
  the served part), the ids stay the sampled ones, and the witness carries
  the text so continuation matches the answer with it. Only vllmx `VLLM`
  serves it (the proxy passes it through, replay re-emits it); every other
  backend refuses the call. A traced step stays token-faithful (the input
  ids end with the injection, the output ids are the sampled continuation,
  the roundtrip re-renders `ids(template) + ids(text)`) and is marked —
  `extra.response_injection` on the step, `extra.response_injected` on the
  trace — because its messages do not carry the response's opening: a
  consumer that re-tokenises text (teacher→student SFT) must refuse such a
  trace; one that trains on ids needs nothing.
- **Answer sampling** (`GenericSamplingParameters.after_reasoning`, an
  `AnswerSampling`): the answer — what the model writes once its reasoning
  closes — sampled with its own `temperature`, `top_p`, `top_k` and `min_p`
  (the last two as `extra_params`, merged over the call's key by key). Any
  other key stays the call's and is refused on the answer: the cap, stops,
  seed and a minimum bound the whole reply, a ban is checked on the joined
  reply, and presence and frequency penalties count a request's own output,
  which in the answer request leaves the reasoning out. A call whose own
  stops end the reply at the reasoning's end (the token or its text) is
  refused: it asks for no answer to sample. vllmx `VLLM` reads where the
  answer starts off the prompt ids with the reasoning parser: a prompt that
  already closed the reasoning (a template rendering no thinking, an
  injection closing it) is one request at the answer's settings; one that
  leaves it open is two, the reasoning stopped at its end token and the
  answer sent the same ids plus the reasoning, with the cap the reasoning
  left, on the same prefix-cache thread (its decode tail resumes it within
  one `prefix_match_unit`). The two stream as one reply: the reasoning's
  prompt, both outputs' ids and text with the end token's text restored
  (vLLM leaves a stop token out of the text), and the reasoning's cache hits
  less what the answer request prefilled again, so a resume that missed the
  cache reads as a `[recompute]`. A reasoning that spends the whole cap on
  reaching its end is reported capped, as one request would be. Settings
  that change nothing are one request. A call carrying `min_tokens` is
  refused where the reply splits: vLLM masks every stop token until a
  request's minimum, the reasoning's end included, and the answer request
  would start the minimum again. Only vllmx `VLLM` serves it (the
  proxy passes it through, replay compares it); every other backend refuses
  the call, and so does a traced one, whose step records one sampling.
  Two requests rather than a switch inside the engine, so it needs nothing
  from the vLLM fork; the cost is a second trip through the engine's
  waiting queue per split turn, and a sampler switch in the fork is the
  one-request route (`roadmap.md`). A logits processor cannot carry it:
  under speculative decoding the rejection sampler skips custom processors.
- **Enforced invariants**: a non-empty `final_content` unless
  `stop_reason == "max_tokens"` (else `MalformedOutputError`, retried
  by the manager) — or, on a backend configured with
  `accept_malformed_output`, the empty reply is kept as the model's turn
  with `stop_reason == "malformed"` and the reason on `malformed_error`,
  traced with `extra["malformed"]` on both trace-record kinds; and
  `token_usage` populated on every response.
- **Token continuation** (`continuation.py`): a capable backend always
  leaves a `ServedTurn` witness on
  every processed answer (the ids it was served after and sampled, the
  template's render of the answered history) and, on the next call, sends
  `served ids + the tokenised delta` — never a re-tokenised history. The
  witness chain stores every served token once; `Conversation.history_epoch`
  is the agent's declaration of a rewrite, after which the next call
  renders afresh. `exact_history` turns every other reason to render
  afresh into `HistoryDriftError`. vllmx serves the id prompt to the
  engine; `OpenAIServer` sends it on `/v1/completions` and lifts the answer
  into the chat shape (vLLM server only, non-streaming — the ids and the
  tokenizer identity come from the `trace_capability()` probe, which the
  host runs at setup for exact history whether or not it traces). Trace
  steps on a continued call carry an echo witness instead of the
  `/tokenize` roundtrip. Two guarantees close the loop
  (`test_token_native_serving.py`): every continued call is checked
  against the engine's own account — echoed ids and billed
  `prompt_tokens` must equal the sent id list exactly
  (`check_continuation_served`, raised never absorbed) — and the delta is
  tokenized once even when the request is first weighed (`token_count`
  answers the id list `respond` then sends; the one-slot
  `_continuation_cache` shares the encode between them).

  A remote engine client verifies the complete serialized fast-tokenizer
  fingerprint, chat template and special-token configuration against the
  serving engine before rendering. Response injection is encoded separately
  and appended to the rendered generation prefix; the trace reconstructs
  that same recipe. Injected tokens belong to the prompt, never the sampled
  completion. The agent decides which fresh contexts receive an injection.

  **Where exact history cannot hold**, and `exact_history` must stay off
  (a capable backend still continues wherever it can; these are fresh
  renders, never drift):
  - a hosted API (OpenAI, OpenRouter, Anthropic): no token ids out, no id
    prompt in, no tokenizer identity — the setup gate refuses;
  - `OpenAIServer` with `streaming=True`: the SDK accumulator drops the id
    extras — refused at setup;
  - `ReplayBackend` / stubs: no served ids at all — refused at setup;
  - an image in the prompt over the server route (`/v1/completions` has
    no image input) — the first image call raises; the in-process engine
    continues it, expanding the delta's placeholders through its processor
    with the images that arrived since the last answer;
  - `send_back_reasoning` other than `"all"`: stripping past reasoning
    re-renders the answers the model was served — the first call raises;
  - `VLLMHarmony`: no witness yet (roadmap) — refused at setup.
- **Capabilities beyond respond()**: `token_count()` — exact token
  counting or `None` for "cannot tokenize" (implemented by vllmx
  `VLLM` from its own engine input, `OpenAIServer` by asking the
  server's `/tokenize`, `ProxyBackend` by RPC, `ReplayBackend` from
  recorded counts); `engine_metrics()` — the scheduler numbers behind
  this backend, or `None` when there is no engine to report (a hosted
  API, a stub, or a proxy client whose host reports instead);
  `trace_capability()` /
  `_build_trace_step()` — the R5 trace seam (tracing.py);
  `capture_sink` — capture of every call (capture.py); `status_hook` —
  host-set liveness scope around the generation wait.
- **Errors**: `MalformedOutputError` (parse/validation failures;
  `DegenerateOutputError` subclass for `!!!!!`-spam with a jittered
  sleep so a solver fleet doesn't reconverge on a poisoned batch) and
  `PromptTooLongError`, which each backend translates from its
  provider's context-length error strings.
- **Copy semantics**: backends deepcopy AND pickle as configuration
  clones (both route through `__getstate__`) — transient live resources
  are dropped and rebuilt on first use, per-process accounting resets,
  `trace_recorder`/`status_hook` are nulled, in-flight requests assert,
  and direct-engine backends refuse to pickle (reach them via the
  proxy). A deepcopy is a detached clone, never a share; code that
  wants to keep the shared instance through a surrounding deepcopy says
  so by seeding the memo (`copy.deepcopy(obj, {id(backend): backend})`).
- **Accounting**: `TokenUsage` (+ pricing, incl. the auto-applied
  Anthropic table), `total_usage`, a `_request_log` behind
  `diagnostics=True`, and `warning_log` — the defense audit trail:
  `tool_error`, `malformed_output_retry` / `malformed_output_recovered`,
  `prompt_too_long`, `truncated_tool_calls`, `tool_output_truncated`.
  When a cancelled stream can report partial usage, that usage remains on
  its incomplete response element as well as joining `total_usage`, so a
  turn observer can attribute it without a backend-wide counter delta.

## Managers (manager.py)

- **`ResponseManager`** runs one LLM call with the malformed-output
  retry loop (default 5): rollback, seed bump, and an invisible-newline
  perturbation of the last perturbable content (defeats deterministic
  kernel-level failures that survive a seed change), all restored on
  success; `PromptTooLongError` becomes a synthetic
  `do_not_send_to_agent` response instead of an exception. It passes no
  `run_id` — trace attribution is structural on `ProxyBackend.run_id`.
- **`TurnManager`** runs a full turn: tool-call loop (parallel by
  default, serial in call order under `stop_when`), per-turn completion
  cap (`max_generated_tokens_per_turn` — lowers `max_tokens` per round,
  synthesizes a capped placeholder when exhausted), truncated-tool-call
  detection (the turn ends and the calls are left UNANSWERED — a known
  protocol wart, kept while committed references still record it),
  and caller hooks, among them `before_round`
  (runs before every LLM call — the just-in-time conversation-reshaping
  seam, e.g. the duck's eviction), `stop_when` (ends the turn after a
  tool result; skipped calls get error `ToolResult`s so nothing
  dangles) and `after_tool_result` (replaces each handled call's result,
  a refusal included, before the turn records it — e.g. the seal's
  suggestion to move). Protected append/discard observers let an instrumented
  subclass retain turn-owned accounting even when a generated round is
  removed. Plain nested managers forward to context-local observers;
  instrumented children override and own their activity instead, so
  overlapping tasks neither lose nor duplicate usage. `live_snapshot()`
  is the side-effect-free progress dict for live viewers. Tool warnings
  drain into `backend.log_warning`.

## Capture, tracing, introspection

- **capture.py** — capture/replay of backend traffic. `CaptureSink`
  writes one record per call (canonical request + outcome, including
  malformed and prompt-too-long paths, plus `token_count` calls) to
  append-only `records.jsonl` + content-hashed `blocks.jsonl`.
  `ReplayBackend` serves a stream back in ordinal order — call N must
  be record N — verifying each incoming request as canonical JSON
  (`ReplayMismatchError` with a readable diff) and re-emitting the
  recorded outcome. Identity-blind by design: which stream a backend
  owns is expressed by where its directory points. `_model_contents` is
  never captured.
- **tracing.py** — the R5 trace-step production surface:
  `TraceCapability` (frozen serving-stack identity),
  `TraceStepRecord`, and the reconstruction helpers. Witness token ids
  always come from the engine; the input-token roundtrip is a hard gate
  (mismatch stops the run), output re-rendering is recorded evidence.
  The host-side collector lives in `platypus.tracing`. Under an ids-only
  trace (`Backend.trace_ids_only`, TAAF R5.39) a step keeps the ids, the
  images and the sampling, and pays no roundtrip, decode or messages copy;
  a continued prompt's echo is still checked. Only the in-process engine
  produces them (`TraceCapability.contents`); a served backend asked for
  ids-only traces is refused at setup.
- **traffic.py** — the traffic benchmark (`traffic_benchmark.md`): recorded
  engine traffic replayed against a live engine. A workload is cut from a
  recording's calls at a snapshot moment, as lanes (the chains of sequences
  one slot served in turn, linked by the collector's slot provenance) that
  start from the history each held there. `replay` plays the lanes
  closed-loop through `VLLM.serve_ids` from a cold engine — each prompt
  carrying the lane's own replies where the recording shared a prefix, each
  reply forced to its recorded length — and counts the tokens streamed
  inside a fixed window past a settle. It has no scheduler: the recording's
  slot decisions are replayed as they fell.
- **introspect.py** — read-only summaries over finished conversations
  (`ResponseSummary`, `turn_response_summaries`) so hosts never walk
  `conv` internals; carries the I1 accounting convention
  (`uncached_input_tokens` = prompt minus cached).
- **engine_monitor.py** — periodic vLLM engine-status line, read from a
  backend's `engine_metrics()` (in-process prometheus snapshot, or a
  server's `/metrics`); counter-derived figures are per-interval deltas;
  never fatal. Local and connector prefix-cache hit rates are reported
  separately (`prefix_hit` / `ext_hit`), since the external one is
  measured against what the local cache missed. `BackendFactory` owns the
  monitor's lifetime for every factory — including one that merely points
  at a server somebody else started — and probes once, so a backend with
  no engine stays silent. `watch_engine_exits` logs how each of an
  in-process engine's core processes ended (a signal by name, or an exit
  code), which vLLM does not; `EngineFactory` starts it with the engine.
- **html_render.py** — semantic HTML for a serialized conversation,
  shared by diagnostics and capture viewing; presentation-only,
  `<details>`-based, no scripts.

## Backends (backends/)

- **common.py** — `to_message_dicts(...)`: conversation element →
  OpenAI-style message dicts, parameterized by assistant-text
  extractors (with/without reasoning, native tool-call markup
  preserved), `reasoning_field` (past reasoning as an
  assistant-message field — the vLLM-server round-trip), and
  `raw_tool_calls` (tool calls stay in text, results become user
  messages — Qwen XML templates). Skips `do_not_send_to_agent`
  elements. Three deliberately-duplicated multimodal user-content
  walkers exist (OAI / Anthropic / HF-processor variants — same walk,
  different image encoding; factor only if a fourth appears).
  Image-carrying `ToolResult`s and image *output* are out of scope and
  fail loudly.
- **vllmx.py** — shared rendering over an in-process or remote engine. `VLLM`:
  HF chat templates, `reasoning_parser` split, `tool_call_parser`,
  `chat_template_kwargs` (`enable_thinking` defaults True; a request the
  native effort path renders without thinking is split by a parser built
  for that, since a think-style parser reads a closing-tag-less output by
  its construction-time setting),
  multimodal via `AutoProcessor` with vision-token placeholders, the
  processor built as vLLM builds the engine's (the stated
  `mm_processor_kwargs` a processor does not take per call go to its
  construction, which is where a Qwen processor honours a lone
  `max_pixels`), and an image the processor would resize DOWN raises
  rather than reaching the model smaller than it was rendered (a
  processor shrinks past its pixel ceiling silently; which stated ceiling
  holds depends on that construction/call split, so the check hands the
  processor a blank image of each rendered size and compares what comes
  back with its resize rule's rounding alone, and an image under the
  floor is enlarged and passes);
  `_prepare_input` returns `{"engine": ..., "trace_ctx": ...}`;
  past reasoning rides the template's structured `reasoning_content`
  field with reasoning-free content (the template owns the think
  block); reasoning tokens counted via a fallback for templates that
  pre-fill `<think>`; exact `_token_count`. `VLLMHarmony`: openai-harmony
  rendering/parsing by channel; no multimodal. Both: context-length →
  `PromptTooLongError` translation, degenerate-output detection with
  engine-side request abort (disabled by `accept_malformed_output`). `VLLM`
  holds an `EngineLike`; only a `RemoteEngineLike` may cross pickle, with
  parsers rebuilt lazily. In-process engines, including `VLLMHarmony`,
  must stay behind a backend proxy. `patch_engine_core_sigint_for_jupyter()`
  keeps engines alive across notebook interrupts.
  `require_explicit_sampling` rejects missing temperature/top-p and extra kwargs
  shadowing named sampling fields, both during preparation and before generation.
  It does not prescribe an RL distribution: the trainer must independently check
  the effective penalties, filters, limits and other settings. `lora_request`
  selects an adapter per request; every engine output must echo that identity.
  Trace capability names the base. Every step has the same `extra["adapter"]`
  schema: `name`, `id`, `path`, `server_instance_id`. Base weights use the
  model name, ID 0, and null path; in-process engines use a null server ID.
  Remote steps retain the serving instance ID. The serving owner must drain
  requests before changing weights or retiring adapters; changing a field does not synchronize an engine.
- **engine_interface.py / engine_proxy.py** — `EngineLike` defines the
  generate/abort/tokenizer/model-configuration boundary used by `VLLM`.
  `RemoteEngineLike` explicitly adds connection and close lifecycle, metadata,
  default LoRA selection, replay policy and the `request_context` envelope.
  `EngineProxy` implements it over TCP HTTP or a Unix socket, using a
  verified local tokenizer. Tokenizer remote code is disabled by default;
  enabling it also requires a caller-chosen `tokenizer_path`, since loaded
  tokenizer code executes in the client process. Nonempty context is refused
  before generation on an ordinary engine; application metadata must not leak
  into vLLM kwargs. The server forwards opaque `request_context`
  metadata to an application-owned engine wrapper, separately from decoding
  parameters. A request ID identifies one generation; transport never
  resends it. Cancellation and close drain outstanding requests within a
  bounded deadline: the server declares `max_generation_seconds` (900 by
  default), and the client's `generation_timeout_s` may shorten it. Duplicate
  request IDs remain reserved for the server lifetime; the bounded ledger
  refuses new admission when full rather than evicting IDs and allowing
  replays. Any uncertain or invalid engine attempt poisons that
  proxy: its owner must stop the run, even if a game catches the error. A
  transport failure cannot prove that no draw occurred, so another request
  on the same failed proxy would compromise single-attempt accounting.
  Applications own admission, policy updates, capture and run-level errors.
  Training's independent history, compaction and score checks live in its
  engine wrapper, not in the generic transport.

  `VLLM.verify_engine_output` verifies final prompt/request identity and
  requested sampled logprobs, then replays the engine's incremental
  detokenizer before parsing. Text must match the sampled IDs, including
  stop-token exclusion and buffered incomplete UTF-8. It requires
  `skip_special_tokens=False` so parsing sees reasoning and tool markers.
  The remote path advertises `platypus.vllmx.engine_proxy_roundtrip.v1`; the
  local path advertises `platypus.vllmx.inprocess_roundtrip.v1`.
  Accepted `ValueError`s from reasoning/tool extraction retain the engine's
  cap signal and sampled continuation. Other exceptions, including failures
  in the surrounding pipeline, propagate. History uses the same acceptance
  policy. Repetition alone is unmarked when malformed output is accepted.
  `request_context_provider` is live
  host wiring: it receives the run ID and previous completed request ID,
  runs only for generation, and is omitted when the backend is pickled.

  `token_count_evidence=True` returns an integer-compatible `TokenCount`
  carrying `prompt_token_ids`, the exact text prompt measured by the same
  renderer used for generation. The ARC3-Training gateway reads those IDs
  to attest the candidate prompt when admitting a compaction reset, rather
  than accepting a harness-reported count alone. This is prompt evidence,
  not proof of generation or permission to reset: the trainer owns lineage,
  context-limit and compaction checks and compares against actual requests.
  Counts crossing `ProxyBackend` preserve this evidence through pickle;
  ordinary counting keeps an integer payload, and multimodal counts
  do not carry prompt IDs.

- **proxy.py** — a WHOLE backend served over a Unix socket. The client
  (`ProxyBackend`) is importable without the host's libraries, keeps
  its own accounting, ships user-format conversations (model contents
  cross as opaque bytes the host unwraps — reasoning round-tripping
  survives), carries `run_id` per game for trace attribution, and
  forwards `token_count` as its own wire op. Capture is client-side:
  a `ProxyBackend` records what it ships, not the host's rendering.
  Cancellation unwinds via an ABORT frame. The HELLO handshake advertises
  the backend's `allows_generation_replay()` capability before any RUN.
  Hosts may disable replay with `replay_on_link_loss=False`; `EngineProxy`
  always refuses generation replay, including before the first response
  arrives; the failure surfaces immediately instead of sampling
  again. This policy propagates through nested proxies and applies to the
  ordinary `LLMAgentSolver` client without a trainer-specific subclass.
  Other backends retain their existing reconnect behavior. A client may
  additionally opt out with `replay_on_link_loss=False`; it cannot override
  a host's refusal. Token-count calls never generate or authorize a replay.
- **oai.py** — `OpenAIServer` for any OpenAI-compatible API. Reasoning:
  native effort or `<think>`-tag parsing; `send_back_reasoning`
  requires `parse_think_tags` OR `reasoning_history_field`.
  `chat_template_kwargs` rides `extra_body`. Defenses: tool-call
  salvage from `<tool_call>` markup when the server parser misses,
  streaming bad-JSON tool-args snapshot recovery,
  `LengthFinishReasonError` → `max_tokens` stop, `"strict": True` on
  tool schemas (a client-side SDK requirement), context-length →
  `PromptTooLongError`. A transient provider failure is retried until the
  first snapshot, up to `max_transient_retries` or, where set, for
  `transient_retry_window_s` since the call's first attempt; each retry's
  line carries the provider's message and headers, how long the attempt
  and the call have been failing, and how many other requests the backend
  had open. Cache-control breakpoints; exact counting by
  POSTing the prepared messages to the server's `/tokenize` (`None` for
  an endpoint without that route, which is the contract's "cannot
  tokenize" — never an estimate dressed as a count); full trace support
  (identity from `/v1/models` + runtime-export sha, per-step adapter
  drift guard).

  Counting deliberately does NOT re-render the chat template locally:
  that means reproducing the template, the processor's image expansion
  and the shape the template wants tool-call arguments in — and the wire
  format disagrees with the template on the last of those (arguments
  travel as a JSON string, the template reads them as a mapping), so a
  local copy is a second implementation that drifts or fails.
- **kimi.py** — `KimiServer`, `OpenAIServer` on Moonshot's Kimi API, the
  distillation teacher's backend (`tools/teachers/`): reasoning parsed
  from and replayed as `reasoning_content`, the API's fixed sampling
  refused rather than overridden, extensions it would ignore silently
  refused, an empty past reply sent as `(no text)`, the prompt cache's
  lifetime asked for, a twelve-hour retry window, and on a traced call
  the top 20 per-token alternatives kept in the step's
  `extra["teacher_logprobs"]`, held as `tracing.PackedJson` until the
  trace is written. Its docstring lists the API's differences as
  measured.
- **anthropic.py** — native Anthropic shapes: tool results as content
  blocks in synthetic user messages, thinking blocks with verbatim
  signatures, same-role merging, multimodal image blocks,
  cache-control breakpoints, adaptive thinking (forces temperature
  1.0), pricing auto-populated from the model prefix, stop-reason
  normalization, prompt-too-long translation.
- **stub.py** — `StubBackend` replays a scripted sequence of
  `LLMResponse`s (fills stop reasons, usage estimates, call ids;
  `from_script` / `from_conversation`; `on_exhausted` policies) — the
  deterministic driver for manager/tool-loop tests.
  `ArithmeticStubBackend` is the request-dependent variant (answers the
  last `a+b` in the final user prompt). **user.py** — a human plays the
  LLM over stdin.
- **config_local.py / victim_offload.py** — engine-side configuration
  for the in-process backends, and a host KV tier behind the
  GPU cache: a vLLM offloading connector that copies a block only as
  the pool is about to lose it, ranks a recurrent model's checkpoints
  by what a lookup will ask for, and checks every copy against the hash
  it was read under. Off unless a solver asks for it; rationale,
  defaults and measurement in `kv_offload.md`.

## Tools (tools/)

`Tool.auto_correct_arguments` controls argument repair before schema validation
and execution: single-argument name remapping, `_raw` JSON-object unwrapping,
and tool-specific escape heuristics. With repair disabled, decoded arguments
are validated unchanged; malformed calls return ordinary tool errors so the
model can correct them on a later turn. Disabling repairs does not disable schema validation.
Write-style tools share `Tool._repaired_text`; Python's escape repair additionally
requires the repaired source to parse. Bound Python views and linked file tools
inherit the target tool's repair policy.

- **python.py** — `PythonTool`: a subprocess Jupyter-style REPL drawing
  slots from a machine-wide `KernelPool` (cross-process
  `SharedSemaphore`s named from a configurable `name_prefix`; a
  module-level default pool serves tools constructed without one). Two
  caps, measuring different things: `max_running` (24) bounds what is
  *executing*, while `max_kernels` (1024) bounds what is *alive* — a slot
  taken on a tool's first cell and held until teardown, so it spans the
  owner's every idle moment in between. The second is set far above any
  real concurrency so a caller's own cap is what gates it. Both waits are
  reported through `status_hook` and are taken *before* the execution
  deadline starts, so queueing is never charged to the code being run. The
  execution timeout is a wall-clock deadline over the whole call —
  host-call interludes don't renew it, servicing a host call is never
  interrupted, and `host_reply_grace` bounds one post-deadline reply
  window; enforcement is SIGINT → SIGKILL. Worker hardening: rlimits
  (address space, FSIZE, NOFILE), nice, optional network-namespace isolation,
  `<repl>` traceback shaping that hides framework frames. Output is
  truncated by characters (head+tail) with a warning pushed to the
  manager. **Host functions are model-reachable by contract**: validate
  host-side and never close over secrets. Unknown dispatcher targets are
  refused recoverably; a registered host function raising is FATAL
  (`pending_host_exception`). Lifecycle: `reset()` recycles the kernel
  (keeps tmpdir + slot; ~40 ms — the sanctioned A4 cadence op),
  `clear_state()` is final teardown, `is_fresh` signals bootstrap,
  `bound(host_functions=...)` gives per-use views, and
  `linked_file_tools()` scopes Read/Write/Edit to the tmpdir at call
  time. Deepcopy/pickle preserve tmpdir *contents* at a new path and
  never the kernel; pickle hard-fails on any live kernel (the snapshot
  path can never silently drop kernel state). Protocol corruption
  (non-JSON worker output) kills the kernel and returns the text as an
  error; a debugger-attach subprocess workaround wraps kernel spawn.
  Per-execute records (`pop_execution_records`) feed diagnostics.
- **file.py** — `FileTool` base with `allowed_files`, `path_mapping`,
  `allowed_dir` (+ `allowed_suffixes`), and `linked_tool` scoping;
  Write/Edit run double-escape heuristics when argument repair is enabled — Edit
  requires a unique verbatim match.
- **subagent.py** — wraps a prepared TurnManager as a callable tool;
  used conversations are parked for diagnostics and the manager is
  restored from a snapshot.

## Reasoning effort and passback

`SystemPrompt.reasoning_effort` expresses intent; backends implement it
natively where possible (OpenAI param, Anthropic adaptive thinking,
Harmony) and otherwise fall back to an optional per-backend prompt
snippet (`reasoning_effort_prompts`, empty by default). Providers
without a `"max"` level map it to `"high"`. Reasoning *passback* is the
resolved `send_back_reasoning` policy — per-call override, else the
backend default `"current_turn"`; `"all"` is the Qwen/vLLM mode and on
`OpenAIServer` needs think-tag parsing or the reasoning history field.
