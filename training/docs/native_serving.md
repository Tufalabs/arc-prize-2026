# Native vLLM serving

The trainer launches `training.native_vllm_server` through
`scripts/launch_platypus_server.sh`. It calls vLLM's `AsyncLLMEngine` directly:
0.25.1 for the 27B, the harness lock's fork (`arc-prize-2026-vllm@3bda1ba09`) for Flash. The harness connects to the trainer's capture gateway, which validates
the sampling policy and token-history lineage before forwarding a draw.
There is no OpenAI-compatible serving endpoint or server-side tool parser.

The harness uses ordinary `VLLM` with a picklable `EngineProxy`, so local and
remote engines share rendering, parsing, continuation and trace construction.
`scripts/training_backend.py::configure_training_agent` composes public Duck
controls with the trainer's distribution, traces and zero request gap. Prompt
injection (the served opening and its schedule) is not set by training: the
Duck preset for the engine decides (the Flash preset serves none, the 27B its
own opener), the worker records the resolved values with every rollout and the
trainer pins them in the checkpoint identity, so a changed preset refuses a
window or a resume. Injected text is prompt input, excluded from the
generated-token training loss. Each worker explicitly opts out of deriving
limits for its supplied engine; Duck verifies the connected engine's window
before any games start. Game children keep the existing CPU-only per-game
backend proxy.

Two more departures from the deployed Duck are deliberate. Malformed replies
are kept, never resampled: the ARC-2 worker sets `malformed_retries=0` (the
Duck's default is 3), no retry perturbation or malformed-reply nudge,
`preserve_failed_draws` and `classify_tool_calls`, because every sampled draw is
a step the learner trains on and the reward charges per malformed call, an
empty reply one, up to the episode cap ([arc2_training.md](arc2_training.md)
gives the count and the trainer's re-derivation of it). `configure_training_agent` turns on
`require_exact_context` and `compact_before_request_only` (both off in the
Duck): the gateway's lineage needs each request's exact token count and
history, where the Duck's defaults may fall back to a bound or trim between
requests. The error policy is the deployment's: workers run with
`on_agent_error="crash_game"`, platypus's default, so an agent exception ends
only its own game, which Benchmark teardown marks `crashed` and scores on what
it reached, and the window goes on. Until 2026-10-11 they ran `"raise"`, which
ended a run on one Duck bug in one game ([arc2_training.md](arc2_training.md)).
A serving failure still stops the window: the gateway poisons it on any failed
generation.

The trainer owns both server processes, the engine wrapper, admission ledger,
score reconciliation, runtime attestation and fatal-error callback. Platypus
supplies generic engine transport and model mechanics. Matched-policy
trainer/inference logprob checks still run on the trainer against captured
sampled tokens before optimization.

Configure the server as an HTTP origin, for example `http://127.0.0.1:1234`.
Paths such as `/v1` are rejected. Existing `SERVER_*` model, revision, device,
LoRA, attention, and cache settings still configure the launcher. The model's
generation configuration is disabled and logprobs are the processed ones (with
the admitted logit bias). Runtime attestation forbids CPU KV offloading and
every host-resident weight but Flash's Engram PLE table, and admits speculative
decoding only on Flash, with exact verification and the drafter the run's config
names. training/FLASH_RL.md ("Serving") lists each Flash engine setting and how
it differs from the harness's own deployment.

Training and evaluation collectors require an explicitly loaded LoRA adapter.
The trainer exports its initialized adapter as policy -1 and publishes it
through `initial_policy.json` before collection starts. A fresh model uses
identity LoRA (zero initial weight delta), exercising the same serving path as
later updates; warm starts export the loaded policy. The standalone
`eval_base_vs_lora.py` comparison still permits its explicit base-model arm.

`POST /generate` uses the shared `engine-proxy-v1` envelope:

```json
{
  "model": "policy",
  "request_id": "run-id#unique-nonce",
  "prompt": {"type": "token", "prompt_token_ids": [1, 2]},
  "sampling_params": {
    "temperature": 1.0, "top_p": 1.0, "top_k": -1,
    "max_tokens": 4096, "presence_penalty": 0.0,
    "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    "min_p": 0.0, "min_tokens": 0, "n": 1,
    "ignore_eos": false, "skip_special_tokens": false, "logprobs": 0
  },
  "lora_request": {
    "lora_name": "policy", "lora_int_id": 1, "lora_path": "/published/adapter"
  },
  "request_context": {
    "arc3_lineage": {"version": 1, "epoch": 0, "kind": "start", "reason": null},
    "arc3_progress": {"version": 1, "after_nonce": null, "score": 0.0,
                      "stage": 0, "data": {"decision_elapsed_seconds": 0.0}}
  }
}
```

A per-request seed is optional. The trainer validates the exact sampling
allowlist, configured policy identity and task metadata before dispatch. It
stores lineage and progress and removes `request_context` before the engine
hop. Transport code treats this context as opaque; it contains no ARC rules.

The only logit bias admitted is a summary's ban on `<tool_call>`. The gateway
requires exactly -100 on the configured `agent_summary_forbidden_token_ids`,
on over-trigger requests only. The server takes its own list from the launcher
(`--summary-forbidden-token-ids`, `SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS`, default
248058, the Duck's `QWEN_TOOL_CALL_TOKEN_ID`), refuses to start unless each id
decodes to `<tool_call>` under the served tokenizer, and refuses a request that
biases any other id. The id is a literal on both sides; the startup decode is
what would notice a tokenizer that moved it.

The trainer discards the harness fork's `prefix_cache_context` and
`prefix_cache_epoch` hints in `sampling_params.extra_args` before sampling
admission and forwarding. Its qualified serving stack owns cache behavior;
other `extra_args` fields are rejected rather than silently ignored.

The response carries the native `RequestOutput` structure: `request_id`,
`prompt_token_ids`, one entry in `outputs`, `finished`, `num_cached_tokens`,
`lora_request` and `adapter`. The output entry includes exact `token_ids`,
per-token `logprobs` rows, engine `text`, `finish_reason` and `stop_reason`.
The gateway extracts the sampled token's raw logprob from each row. These
values come from the actual engine output; text is never retokenized to
reconstruct training evidence. Malformed tool syntax and repeated characters
remain ordinary output. A disconnected caller's admitted request drains to a
definite outcome without another generation. Inconsistent engine evidence or
transport failure permanently stops training.

`text` is the engine's incremental detokenizer output. It may omit a final stop
token or differ from decoding an isolated token suffix, so the client preserves
it for parsing. The trainer-owned engine wrapper and shared client both verify
it against the same incremental detokenizer semantics, including excluded stop
tokens and buffered UTF-8. The engine wrapper also checks unconsumed terminal
draws before capture. Other text/token mismatches are fatal; decoding an
isolated suffix is not the validation rule.
The complete sampled IDs and logprobs remain authoritative for training and
continuation, including the terminal token. Output IDs are bounded by the model
vocabulary; input admission also permits the tokenizer's maximum ID, matching
vLLM's handling of model-only and sparse tokenizer IDs.

The service exposes the routes below. The harness gateway permits read-only
metadata, model discovery, health, runtime attestation and metrics. It blocks
all adapter controls, which are called by the trainer directly:

| Route | Purpose |
| --- | --- |
| `GET /health` | Check native engine health. |
| `GET /runtime` | Attest actual sampling and offload configuration. |
| `GET /metadata?model=policy` | Tokenizer/model metadata and selected adapter identity. |
| `GET /models` | Current aliases, identities, and engine context limit. |
| `GET /metrics` | Native engine Prometheus counters, including prefix-cache usage. |
| `POST /adapters/load` | Load `{"name":"policy","path":"/absolute/export"}`; returns its identity and LoRA census. |
| `POST /adapters/unload` | Remove `{"name":"policy"}`. |
| `POST /adapters/warm` | Run a trainer-only one-token warm-up for `{"name":"policy"}` (or the base name). |
| `GET /weights/info` | NCCL weight-transfer engines (27B `full_weight`, Flash): this engine's workers in the group. |
| `POST /weights/init`, `/pause`, `/start`, `/update`, `/finish`, `/resume` | Join the trainer's NCCL group once, then publish: drain, start a publication, receive it from the trainer's rank 0, finish (Flash: verify), serve it. |
| `GET /policy_weights/info` | Flash: the published policy's `sha256` and `iteration` (null while the checkpoint's weights are served). |

`/metadata` also states `image_max_pixels`, the pixel count above which the
served image processor shrinks an image, or null when images are disabled: the
engine's `mm_processor_kwargs` `max_pixels` when it names one, else the image
processor's `size["longest_edge"]` (the checkpoint's preprocessor config;
16,777,216 for Flash). An image server whose processor states neither does not
start. The launcher sets no `max_pixels`, where the deployed Duck sets it to its
own largest render (`derive_max_pixels`); training turns that derivation off
because the engine is the server's, and neither ceiling shrinks an ARC image. The
value is for the rollout worker's window-start check that the Duck's largest
render (`duck.images.largest_image_pixels`) is at most the served ceiling. It is
not part of `processing`, the ullm processing identity the trainer recomputes
and compares exactly.

An adapter identity has `name`, its actual integer LoRA `id`, `path`, and a
`server_instance_id`. Base-model requests use ID 0 and a null path. Each load
gets a fresh ID, even when reusing an alias. A swap loads the replacement,
retires the previous ID, resets the prefix cache, and then publishes the alias; adapter mutation is
rejected while any generation is in flight. A failed control RPC poisons the
service because its engine-side outcome may be uncertain. Each draw pins its
`LoRARequest`, and the response must identify that same request. The trainer
also fetches the expected identity once per window and compares every returned
identity against it, detecting an unexpected reload or server restart.
The cache reset is required because vLLM 0.25.1 hashes cached prefixes with the
LoRA name, not its integer ID. Unloading also resets that cache. A failed reset
poisons the service before another draw is admitted.

Before the engine sees an adapter, the load takes its LoRA module census
(`training/lora_census.py`). vLLM 0.25.1 checks adapter modules only by name
suffix: an unknown suffix fails the load, but a module with a known suffix whose
path names no LoRA layer of the served model (the vision tower, a layer the model
lacks, the skipped MTP head) is loaded and never applied, and `rank_pattern` and
`alpha_pattern` are dropped (every module is served at `lora_alpha / r`). The
gateway reads `adapter_config.json` and the tensor names of
`adapter_model.safetensors` (the only weights file it accepts), then asks each
engine worker, through the `LoraCensusWorker` extension that
`configure_engine_worker` installs on LoRA engines, under which name vLLM files
each tensor and which LoRA layers the model has, packed layers with the modules
they merge. Every adapter module must be one of those layers or merge into one
(`q_proj` into `qkv_proj`, `in_proj_b` into `in_proj_ba`); otherwise, or with a
nonempty pattern, an unreadable adapter or a module without both LoRA factors,
the load is refused with HTTP 400 and the engine is not touched. A malformed
engine answer poisons the service. The response carries the census next to the
identity:

```json
{"adapter": {"name": "policy", "id": 3, "path": "/published/adapter",
             "server_instance_id": "..."},
 "lora_census": {"applied": {"q_proj": "qkv_proj", "in_proj_b": "in_proj_ba"},
                 "modules": {"q_proj": 16, "in_proj_b": 48}}}
```

`applied` maps each adapter module name to the served layer that applies it and
`modules` counts the adapter's modules of that name. The trainer requires the
census to cover exactly the target modules its runtime export wrote
(`runtime_adapter_summary.json`), and one module per exported A/B tensor pair;
anything else is fatal before the adapter serves a draw. Each publication logs
the census per server.

The `/weights/*` routes serve every engine started with
`--weight-transfer-config '{"backend":"nccl"}'`: on vLLM 0.25.1 they take the
27B `full_weight` publications through vLLM's own update calls; on the Flash
fork they take merged policy weights (training/FLASH_RL.md, "Merged policy
weights"). `/weights/init` joins the trainer's NCCL group once
(`{"init_info": {"master_address", "master_port", "rank_offset", "world_size"}}`,
the trainer's rank 0 is rank 0); `/weights/pause` drains the engine and fences
draws until `/weights/resume`. Flash engines start without LoRA and serve the
policy under the base name. A Flash `/weights/start` carries the label
`{"kind": "policy_weights", "policy_sha256", "policy_iteration"}` and the
publication's manifest (`flash_policy_weights/3`: tensor names in broadcast
order, shapes, per-tensor checksums, iteration, LoRA targets, routers,
rounding); the label must name the manifest's sha256 and iteration, and every
worker checks the manifest against the checkpoint's tensors for those targets
before any byte moves. `/weights/update` (`update_info`: the same names and
shapes, BF16, unpacked) has every worker take each tensor off the trainer's
broadcast and load it through the model's own `load_weights` into the existing
storage, auditing every write; `/weights/finish` recomputes each tensor's
checksum from the served rows, requires the manifest's, and resets the prefix
cache; `/weights/resume` makes it live. From then on the base identity carries
`kind: "policy_weights"`, `policy_sha256` and `policy_iteration`, so every draw
names the publication it was served with. The identity is the sha256 of the
manifest, checksums included, so it pins the served bytes; no publication bytes
touch disk. Any refusal or mismatch poisons the service. Adapter loads are
refused on such engines; there is no route that loads weights from a path.

The HTTP layer requires `SERVER_API_KEY` on all routes when configured. Model
and tokenizer metadata expose only explicit public configuration fields;
credentials and the full engine configuration are not serialized.
The service binds to `127.0.0.1` by default; `SERVER_HOST` can select a network
bind, which requires a nonempty `SERVER_API_KEY` before engine startup.
Surrounding key whitespace is rejected to keep client/server authentication exact.
The service and harness read `SERVER_API_KEY` from their environment; keys
never travel on command arguments or in saved rollout specs.


Launcher readiness probes use the same `SERVER_API_KEY` Bearer authentication
as the trainer. The probe reads the key from the environment, with no key in
its command arguments or output. Single-node launchers record the actual
per-server log files in `inference.server_log_paths` and
`official_eval.server_log_paths`, ordered like their endpoints. These paths
remain under the durable run directory when rollout traces use local scratch.
Custom launchers should supply one absolute readable log path per server.
The trainer reads a log only to quote it in an error: it confirms each adapter
load over HTTP (the load response's identity and LoRA census, the alias in
`/models`, and the warm-up draw served under that identity), so a missing log
never stops training.
