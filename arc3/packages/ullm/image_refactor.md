# ullm image path: processor refactor

*Register: living — proposed design and outstanding checks; update as the checks settle.*

This proposal assumes PR #87 and its review fixes have merged. Use the resulting mainline commit as the comparison baseline. API references below use `Tufalabs/arc-prize-2026-vllm@a1edcd2c`; recheck them against the runtime pins chosen for implementation. Paths refer to arc3-platypus unless prefixed with `ARC3-Training:`.

## Recommendation

Replace ullm's parallel image-processing logic with vLLM's registered processor. Use one prepared representation for serving, counting, and tracing, plus a versioned remote envelope. This should remove duplicated processor setup and the expand-collapse-expand cycle while preserving image delivery, sampled history, and training inputs.

Replace each path only after its tests pass against the merged baseline. Keep the existing implementation wherever the replacement cannot yet meet the same requirements.

Initial scope: still images on explicitly tested decoder-only architectures. Video and additional architectures require separate support.

## Proposed design

### Inputs and ownership

| Object | Contents and rule |
|---|---|
| `RawPrompt` | Unexpanded IDs, ordered image snapshots, effective processing identity/options. Continued IDs preserve sampled tokens as specified below. |
| Prepared request | The raw prompt, a self-contained vLLM `TokensInput` or `MultiModalInput`, and input evidence. Multimodal items contain actual tensors. |
| Media envelope | Versioned wire form: raw IDs, lossless PNGs, processing identity/options, and expected evidence. The server reconstructs its own tensors. |

Local dispatch accepts only validated engine inputs; remote dispatch accepts the wire form. Select the route through an explicit capability after initialization/connection. Replace `hasattr` routing. `TypedDict` annotations need runtime checks: they do not prevent constructing a token input with stray image fields.

Check image/tensor/hash/placeholder counts, ranges, and embedding masks. Preserve `PlaceholderRange.is_embed`, not just offset and length. Reject unmatched placeholders according to the supported model's rules. Text-only requests need no image-specific initialization.

Own a stable copy of each image from preparation through tracing. Do not cache evidence by mutable PIL object identity. A frozen container holding mutable pixels or tensors is not sufficient.

### Processor and cache

Use the registered processor for the resolved serving model. The pinned API is `BaseMultiModalProcessor.apply(ProcessorInputs, TimingContext)`, with image data parsed through the processor's supported path. The implementation is in `vllm/multimodal/processing/processor.py`. Host construction without model weights remains a TODO. [Processor API](https://github.com/Tufalabs/arc-prize-2026-vllm/blob/a1edcd2c/vllm/multimodal/processing/processor.py)

Preparation uses a dedicated, bounded **processor-only cache**, or no cache. Never borrow the engine's sender cache: its hits may omit tensors, and the shared-memory sender returns address items. Reject both forms in prepared requests. Counting or cancelled preparation must not make the engine believe it has received an image. vLLM itself uses a separate processor-only cache for tokenize-only work. [Cache implementations](https://github.com/Tufalabs/arc-prize-2026-vllm/blob/a1edcd2c/vllm/multimodal/cache.py), [renderer](https://github.com/Tufalabs/arc-prize-2026-vllm/blob/a1edcd2c/vllm/renderers/base.py)

Initially submit complete tensors to the engine. Verify this works with the deployed shared-memory and prefix caches; do not change those settings just to make the prototype pass. Keep processing off the event loop, retain serialized access/cancellation draining, and keep processors and caches off the agent objects sent to game subprocesses.

### Options and identity

Merge engine defaults with supported backend overrides at initialization/connection and freeze the effective options for the session. For remote servers with fixed options, omitted or matching overrides are accepted; conflicting overrides are refused. Reconnection repeats identity checks.

Identity includes the logical model and immutable artifact revision/digests, resolved model implementation, relevant effective model configuration, tokenizer/template identity, processor configuration/options, vLLM version/fork commit, hash algorithm, and relevant image/numerical library versions. Exact training also names the position implementation. Retain useful existing identity fields.

Local paths locate artifacts; they do not establish model identity. vLLM hashes include `model_config.model`, so identical pixels loaded under different host/server paths can produce different keys. Establish consistent model identity through a tested configuration mechanism before processing; do not rewrite hashes after cache lookup. [Hash inputs](https://github.com/Tufalabs/arc-prize-2026-vllm/blob/a1edcd2c/vllm/multimodal/processing/inputs.py)

Keep the digests distinct:

- PNG hash: encoded-byte integrity.
- Pixel hash: mode, dimensions, and decoded pixels, including palette semantics where supported; also used for continuation.
- vLLM hash: the engine's processing/cache key.
- Tensor/layout/position hashes: the outputs and placement actually reproduced.

Do not replace pixel checks with caller-supplied UUIDs. Reject identity mismatches with a useful explanation; different runtime versions need an explicit compatibility decision.

### Exact continuation

Only a fresh prompt is tokenized from the full chat render. For continuation, let `R_n` be the raw prompt, `E_n` the expanded prompt served, `O_n` the exact sampled IDs, and `D_n` the newly encoded suffix:

```text
R_(n+1) = R_n + O_n + D_n
E_(n+1) = prepare(R_(n+1), images_(n+1)).prompt_token_ids
E_(n+1) must start with E_n + O_n
```

Keep existing suffix rules for EOS, length stops, reasoning, tool calls, and injection. Include new injection exactly once; preserve earlier injection where it was served. Never decode and re-encode sampled history. Refuse an exact-history request if processing changes its served prefix.

Witnesses need both raw and expanded chains, sampled IDs, epoch, and historical image evidence. Store deltas to keep memory linear. `ServedTurn` currently holds one chain, so it cannot simply be left unchanged.

Check historical pixels, tensors, and placeholder mappings as well as token prefixes. Same-size image substitutions must fail without a declared rewrite. Preserve compaction/rollback epoch rules and their token-count media evidence. Counting and cancelled calls do not advance witnesses.

### Positions and training

`MultiModalInput` does not contain final model positions. Reproducing it is insufficient to prove training parity. In the referenced fork, `get_mrope_input_positions` is an instance method: Qwen3-VL uses configuration, while Flash-Next returns flat positions. Calling an inherited Qwen3 helper for Flash-Next would compute wrong positions. [Qwen3-VL](https://github.com/Tufalabs/arc-prize-2026-vllm/blob/a1edcd2c/vllm/model_executor/models/qwen3_vl.py), [Flash-Next](https://github.com/Tufalabs/arc-prize-2026-vllm/blob/a1edcd2c/vllm/models/qwen4_exp/nvidia/model.py)

Provide an explicit, weight-free position function for each supported architecture. Prefer a function in the fork shared with the actual model method. Any temporary adapter must be checked against the resolved model instance and runner; do not fabricate an uninitialized model or select a helper by name alone.

For exact training:

- Record prompt position hashes and relevant deltas alongside input evidence. Confirm the runner uses those positions, including any later transformations.
- Reproduce positions for the full prompt plus sampled completion using the engine's decode-position rule.
- Define conversion to the trainer's real forward inputs: image fields/order, grids, token types where required, padding, attention masks, and loss masks. Test mixed image/text batches and refuse truncated witnessed image spans.

Preserve the merged baseline's supported training architectures. Extending that set requires position and numerical parity checks. Unsupported training need not disable an independently supported inference deployment.

### Shrink refusal

Keep the current check until a replacement preserves its behavior. It already calls the processor module's own `smart_resize`, distinguishing patch rounding from ceiling-induced shrink. Placeholder length alone cannot make that distinction.

A replacement needs processed geometry and the model's rounding/crop rules. Test floors, ceilings, rounding boundaries, aspect ratios, and inherited options. Do not silently remove this check when geometry is unavailable.

### Serving, counting, and traces

Input evidence covers raw and expanded IDs, identity/options, ordered pixel digests, vLLM hashes, full placeholder ranges/masks, and tensor names/dtypes/shapes/bytes. Serialize other required metadata deterministically. Position evidence is additional.

The server decodes and verifies the envelope, prepares its own input, compares evidence, checks limits against **expanded** length, and serves the same object it checked. Bind returned evidence to the request, returned prompt IDs, server/model, and existing policy/adapter identity. The host verifies it before accepting output or advancing history.

Keep the prompt-ID echo check. Shared host/server code can reproduce the same mistake, so real engine/model tests must independently establish image consumption.

Count from the prepared expanded IDs. Persist enough trace data to reproduce the request in a fresh process: raw IDs, images, options/identity, served expanded IDs, sampled IDs, and evidence. Rendered messages remain a view of continued history. Define when local and remote traces qualify for `media_reconstruction_status="ok"`; an evidence dict's presence alone is insufficient.

Version the wire/evidence format. Reject unsupported versions before generation. Specify how existing traces are read or explicitly rejected; never reinterpret old evidence as satisfying the new contract.

### Performance

Processor caching avoids repeated transformation. PNG encoding/decoding, pixel/tensor hashing, prompt scanning, and transmission may still scale with history length. Do not claim total work scales only with new images.

Measure count, preparation, generation, and tracing together. Include CPU time, queue wait, event-loop delay, memory, payload size, and throughput under cold/warm caches and eviction. Full tensor submission may increase IPC cost.

Add bounded envelope/evidence caches only where measurement justifies them, using verified immutable content and processing identity. Reuse preparation across count and generation only when tokens, images, options, and epoch still match. Set performance budgets from the baseline and deployment requirements before comparing results.

## Implementation order

| Step | Work | Proceed when |
|---|---|---|
| 1 | Record the merged baseline and runtime pins; prototype the processor adapter. No production routing change. | CPU construction, cache isolation, identity, position, and shrink checks pass. |
| 2 | Integrate local preparation, counting, continuation, and tracing. | Real local serving, exact history, and performance checks pass. |
| 3 | Integrate remote transport and training. | Runtime choice, wire rejection, trace reconstruction, and training parity pass. |
| 4 | Delete superseded code and update docs. | Every affected caller uses the verified replacement. |

Before step 3, compare the actual serving and training runtime pins. If they differ, choose a tested migration or a narrowly supported alternative that proves equivalent tensors, layout, positions, and numerical behavior. A matching version simplifies reproduction; it does not prove correct gradients. If neither route is acceptable, retain the existing remote/training path and defer its replacement.

## Validation TODOs

Run the existing image and continuation tests against the replacement and add missing cases below. All refactor checks remain open. Record commits, model/runtime/configuration, command, result, and a durable log/artifact for each item. Skips do not count as passes.

| Check | Required cases and pass condition | Owner |
|---|---|---|
| [ ] Construction | Real registered processors on the intended CPU host, without weights. Verify files/imports, tokenizer/template behavior, model overrides, and placeholder structure. Test inherited, matching, and conflicting options, including construction-sensitive `max_pixels`/`size`. | ullm |
| [ ] Cache isolation | Cold, warm, disabled, evicted, duplicate/reordered images; count-before-generate and cancelled preparation. Tensors/evidence agree and every draw has usable data. Include deployed shared-memory/prefix caches. | ullm + serving |
| [ ] Identity | Same artifacts under different local paths reproduce equivalent inputs/keys. Changed pixels, artifacts, config, tokenizer, options, or unsupported runtime versions are detected. | ullm |
| [ ] Admission | Mutate PIL pixels after capture; swap/remove/add images; corrupt envelopes, ranges/masks, or tensor metadata. Preserve the captured input or refuse before dispatch. | ullm |
| [ ] Continuation | Noncanonical sampled tokens, whitespace, malformed/normalized tools, reasoning, EOS/length stops, injection, multiple image turns, mutation/reordering, compaction/rollback. Exact prefixes and image evidence hold; witnesses grow linearly. | ullm |
| [ ] Positions | Compare the pure function with the actual model method and runner for each supported architecture, including decode positions. Flash-Next is correct or refused. | vLLM + training |
| [ ] Shrink | Real processed grids near floors, ceilings, patch boundaries, and supported aspect-ratio extremes. Preserve accepted rounding/enlargement and reject ceiling shrink. | ullm |
| [ ] Real image use | Merged baseline and candidate agree for first, historical, and newly appended images, including local `exact_history` first calls and declared rewrites. Inspect features reaching the model and use distinct equal-size images in a controlled image-sensitive case. Keep established logprob/logit tolerances. | serving |
| [ ] Wire | Independent host/server caches; tampered IDs, pixels/order, options, identity, layout/masks, or tensor evidence; wrong schema/request/policy. Refuse invalid requests before serving and reject mismatched responses. Check expanded-length limits. | ullm + server |
| [ ] Count and trace | Counted IDs equal served IDs. Reconstruct saved local/remote traces in a fresh process, including compaction media and text-only cases. Corrupt/truncated records cannot claim success. | ullm |
| [ ] Training | Real forward-input conversion with image/text batches, padding/masks, and full completion positions. Run existing same-policy parity and a short real training window with image continuation. Keep existing tolerances; no nesting or infrastructure errors. | ARC3-Training |
| [ ] Load | Compare the whole count/generate/trace cycle at intended concurrency and history sizes, with cold/warm caches and eviction. Report CPU/queue saturation, memory, hashing/PNG costs, payloads, and throughput against budgets fixed beforehand. | serving + training |

## Removing code and merging

Remove processor construction/expansion duplication only after real processor and configuration checks pass. Remove placeholder collapse only after local engine/cache tests pass. Remove `position_inputs` only after position and trainer-forward parity pass. Remove the shrink check only after equivalent behavior passes. Keep echo and integrity checks.

Search all callers, including trace rendering and counting, before deleting helpers. Replace the continuation fingerprint helper with the verified pixel digest; keep engine hashes separate. Do not silently retry a failed exact-training draw through an alternate processor or representation. Rollback is an explicit deployment decision before new draws.

Define producer/consumer pin order and legacy-trace handling across repositories. A wire change from expanded to raw IDs requires a deliberate U5 update in `requirements.md`, called out in the PR. Put new settings in `settings.md` before implementation. Measure reference/golden/snapshot impact before regeneration.

Use `make prepare` during development. For each stage that merges into mainline, run its relevant checks and `make test-full` as the final pre-merge check, following the existing GPU/API and infrastructure workflows.
