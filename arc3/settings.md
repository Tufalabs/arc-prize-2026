# Where a setting lives

*Register: as-built.*

The public harness/trainer boundary is defined in [requirements.md §8,
decision 12](requirements.md#8-decision-ledger). Its placement consequences:
reply caps are agent defaults, stamped on each request's sampling copy;
engine derivations are agent controls, with explicit opt-outs for supplied
factories without engine arguments.

Every setting that determines an agent belongs to exactly one of eight
buckets. The bucket says which package owns it and which table states it, so
"where does this value go" has an answer before the argument starts.

## The two questions

**Is it task-free?** A setting that would be true for anything served by this
engine on this box belongs to `ullm`; one that exists because ARC3 is the task
belongs to `platypus`. The test is a real consumer: the ARC2 repo uses `ullm`
without `platypus`, so anything it needs must be on the `ullm` side.

The coordinates themselves follow that split. `ullm` describes boxes and
checkpoints; each task then says which of them it runs on, and refuses the
rest — `kaggle-4xl4` is a box `ullm` can serve and ARC3 never does.

**Which coordinate decides it?** A deployment is an **engine** (which
checkpoint) and an **environment** (which box, which deploy target). A setting
is decided by neither, by one, or by both.

|  | neither | engine | environment | both |
|---|---|---|---|---|
| **ullm** (task-free) | U-00 | U-E0 | U-0V | U-EV |
| **platypus** (ARC3) | P-00 | P-E0 | P-0V | P-EV |

## Layering

Buckets are precedence, not exclusive ownership. A setting has a **home** —
the most general bucket that can state it — and any more specific bucket may
override it there. U-00 stating a 64 GiB host cache and one cell overriding it
is the normal case.

On the platypus side that reads literally: `duck.presets.configure()` walks
the layers down the page, assigning onto the agent — what ARC3 asks
everywhere, then the engine's, then the environment's, then the cell's. A
later layer overwriting an earlier one is the mechanism, and each chain
refuses a coordinate it does not cover.

General → specific:

    U-00 → U-E0 / U-0V → U-EV → P-00 → P-E0 / P-0V → P-EV

The task layer may override the task-free layer on the same setting, because
ARC3 knows things about its own workload that a checkpoint or a box cannot —
the sampling profile overriding the model card's is exactly that.

**One exception, in the first rung.** The engine-argument corrections in
`deployment.build()`'s U-00 block are written *over* the checkpoint's
arguments rather than under them, so nothing more specific overrides one. They
are there because vLLM's own default is wrong, which makes a family builder
restating one a builder restating a vLLM default, not stating a checkpoint
fact. The rest of U-00 layers normally: the host tier's shipped size is U-00
and the cells that size their own tier override it.

## Derived, not stated

Three engine settings are computed from others at
`LLMAgent.prepare_for_backend()`, the last moment before the backend is built,
so that an override made after construction is seen.

| Derived | From | Why they are one number |
|---|---|---|
| `max_model_len` | `context_window_tokens` under `prune`; `engine_ceiling_tokens` under `summarize` | The agent may build a prompt up to its window and the engine must accept exactly that. The reply reserve is inside the window already. A summary request goes past the window by an amount the agent cannot know, so under `summarize`, the default, the engine keeps the window it was built with, and the agent's own window and reply reserve must leave the summary room under it (`packages/duck/design.md`, "Summarize"). |
| `max_num_seqs` | `run_config.max_concurrent` | The harness admits that many games and each holds one generation at a time. |
| the image processor's ceiling (`mm_processor_kwargs["max_pixels"]`), and no video (`limit_mm_per_prompt["video"]` 0) | `multimodal_upscale`: the largest image the duck sends, frames and both palette legends, each side rounded up as the processor rounds it | The engine's start-up profile reserves activation for one image of the processor's largest size, 16,384 tokens on Flash-Next, and past its ceiling the processor shrinks an image rather than refusing it, so the ceiling follows the render rules. The duck sends no video, so the profile sizes for images alone. On the B200 the peak activation fell from 4.86 to 1.02 GiB (`packages/ullm/flash_optimization.md`, "Competitor tricks"). ARC2's duck derives it the same way, from its own renders, and so does the seal (`SealAgent.derive_max_pixels`), from its own: the frame, the colour-name legend and the index legend in each alphabet a board prints in. |

Each engine argument has a flag on `DuckAgent` (`derive_*`). A value someone
chose explicitly is refused rather than overwritten: the factory records what
it was built with and updates that as the agent settles onto it, so a caller's
number can always be told from one nobody picked.

A setting qualifies only when an agent field that exists **for its own
reasons** has an engine argument that must agree with it. A value the engine
alone consumes has no such counterpart, and copying it through a field the
agent never reads would be a pass-through, not a derivation.

`gpu_memory_utilization` is deliberately **not** in this graph. It is measured
per cell and left a little conservative, and neither derives from
`max_num_seqs` nor drives it.

---

# The buckets

## U-00 · task-free, neither coordinate
*The "neither coordinate" block of `ullm.backends.deployment.build()`, plus
`UNIVERSAL_ENGINE_ENV`, applied to every config the coordinates build.*

Corrections to vLLM defaults that are wrong for every model on every box.
Small by construction — a default that is right for someone belongs to a
coordinate instead. The engine-argument rows are applied over whatever the
checkpoint's builder produced, and are the one place a more specific bucket
may not override (see "Layering").

| Setting | What / why |
|---|---|
| `generation_config="vllm"` | Take vLLM's generation defaults, not the ones shipped inside a checkpoint. |
| `draft_sample_method="probabilistic"` on any speculative config | Not written by the factory into a caller's speculative config: that config is the experiment's, and a key it did not write would be a surprise. The Flash-Next cells, which draft, set it themselves (speculative decoding, below). The recommendation, measured: vLLM's default verifies the head's argmax as one-hot, which caps acceptance at the target's top-1 probability however good the head is; sampled, acceptance is the overlap of head and target (65% against 62.6% on Flash-Next), and the output distribution is the target's either way. Set it wherever a drafter is configured. |
| `prefix_cache_retention_interval=None` | Cache every recurrent state block. The engine's default keeps only boundaries it can prove reachable, and a continuation resumes at its own previous end, which is not one: every continuation would arrive as a divergent hit and recompute state. Every engine here is a hybrid. |
| `mm_processor_cache_gb=0` | No image processor cache. Every request resends each image of its context, and the shared-memory cache evicts oldest first, a hit not refreshing an image, so once the images in use outgrow it the hit rate falls and stays down, while it holds shared memory against the job's cap. The renderer's workers (U-0V) process every image of every request instead (2026-10-17; `packages/seal/evidence.md`, "The engine columns", has the readings). |
| `mm_processor_cache_type="shm"` | For a cell that turns the cache back on: one ref-counted multimodal cache. The default per-process `lru` mirror evicts an item the EngineCore later expects and crashes the request (vllm#31404). |
| `dense_cuda_capture` (a factory field, on) | A captured CUDA graph serves one exact batch width, so a decode batch between rungs of the engine's default ladder (1, 2, 4, then every eighth) is padded up and the model computes filler it discards -- averaging 3.2 sequences, paid in the per-sequence term of every step. On, the factory derives a rung for every width up to `max_num_seqs` and leaves the ladder above alone: decode is never dispatched there, mixed batches are, and prefill overflows past it into no graph at all. Applied at start, so it reads the width however late that was decided; left alone when the width is unset (vLLM's own default differs by serving topology) or when a caller named the rungs. The four-L4 box turns it off (U-0V). |
| `async_scheduling` | Stated rather than inferred. Unset, vLLM enables it unless something is incompatible and then disables it with a warning -- which is how a run quietly loses it. Set, the same incompatibilities raise. Nothing we serve is on that list: it excludes pooling models and every speculative method outside the EAGLE family, which includes MTP. |
| `watermark` | The share of the KV pool kept free when admitting. vLLM's own value is zero, which fills the pool to the last block and makes preemption storms the steady state. |
| `VLLM_USE_V2_MODEL_RUNNER=1` | vLLM's recommended runner, and the only one implementing the recurrent-state blank. Read at engine init, so it goes in the environment before the worker spawns. |
| `enforce_eager=False`, `enable_prefix_caching=True` | CUDA graphs are worth their capture time on anything served long enough to play games, and prefix caching is what makes a block offloadable at all. |
| `MAX_JOBS`, PATH/CUDA_HOME fixups | Build-time parallelism and toolkit discovery for JIT paths. |
| the engine's processes' logging (`config_local.QUIET_ENGINE_LOGGERS` below WARNING) | vLLM's own logging config, handed to the engine's processes through `VLLM_LOGGING_CONFIG_PATH`, without the fork's line for every closed prefix-cache context. During play that line was all the engine wrote to the stdout it inherits, which in a notebook is a pipe one ipykernel thread drains; once that thread stopped, the engine blocked on a write and the run froze (2026-10-29). A config path the caller set is theirs. |
| engine-startup timeouts | Last-resort bounds, not tuned waits: a cold `torch.compile` runs into the tens of minutes. |
| the host tier's shipped size and connector (`DEFAULT_CPU_KV_OFFLOAD_GIB`, `DEFAULT_CPU_KV_OFFLOAD_VICTIM`) | What a tier is when someone turns one on and names nothing. Whether it is on is P-00; how big a run's tier is, P-EV. |
| the tier's engine-log copy (`DEFAULT_ENGINE_LOG`, off) | The connector's instrumentation goes to the job directory's trace; its copy in the engine log is a knob no preset sets, because writing it is what stalled a Kaggle engine for the last two hours of a submission (`packages/ullm/kv_offload.md`, defaults). |
| the tier's park mode, its log and its sibling refresh (`park`, `DEFAULT_PARK_LOG` on, `DEFAULT_SIBLING_REFRESH` `"needed"`) | Park mode is not stated: the connector takes it wherever the engine names each conversation's thread and keeps decode tails (U-E0), and then holds whole conversations on the host between turns. Its log goes to the job directory; TAAF's minimal diagnostics, a submission's, turn it off (`BackendFactory.use_minimal_diagnostics`). A use of a boundary refreshes only the siblings a thread still needs. All three: `packages/ullm/kv_offload.md`, "Park mode". |
| the tier's parking on the GPU (`gpu_park`, on where park mode is; `DEFAULT_GPU_PARK_RESERVE`) | On by default wherever park mode is (2026-10-18), ahead of the run that measures it (`packages/ullm/roadmap.md`). A parked thread stays pinned on the GPU and spills to the host, oldest first, only when the pool needs the room, so a stall that empties the engine parks its threads there rather than overflowing the tier: `packages/ullm/kv_offload.md`, "Parking on the GPU". |
| `profile` (a factory field, an `EngineProfile`; off) | Torch-profiler windows opened on the running engine once its KV pool fills, packed into one tar in the job directory. Off because a window stalls the engine while it closes; a run that wants the kernel breakdown arms it. Its trigger and spacing are the profile's own fields, not any engine's or box's. |
| `memory_log_s` (a factory field; off) | The fork's host and device memory sampler through startup, for a boot that dies on memory. Off everywhere: it was a bring-up instrument. |

## U-E0 · task-free, engine only
*The engine block of `ullm.backends.deployment.build()`.*

Facts about a checkpoint, true wherever it runs.

| Setting | What / why |
|---|---|
| repo id + revision | Identity. The *path* it is mounted at is U-EV. |
| Flash-Next's checkpoint: RadixArk's NVFP4 conversion, 60% of each layer's experts kept (e307) | Provisional, both halves. RadixArk's is the only Flash-Next stack with serving hours on it here and the one the leaderboard scores with; the stock checkpoint is unmeasured on our stack. The prune is for Kaggle's card and is the ratio qualified on the earlier branch; it costs score (one 20-minute pair: 9.02 unpruned against 2–4 pruned). Structured sweeps over both follow; the expert histogram is their instrument. |
| `dtype`, `quantization`, `kv_cache_dtype`, `mamba_ssm_cache_dtype` | The checkpoint's numerics -- and for Flash-Next the recurrent state in bf16 where the checkpoint says float32, placed with the KV cache's format by decision (2026-09-21): both set what the cache holds, and both were validated on score (`packages/ullm/flash_optimization.md`, "The recurrent state in bf16"). |
| `reasoning_parser`, `tool_call_parser` | What its output needs parsing with. |
| `chat_template_kwargs`, `native_reasoning_effort`, the effort scale | What its template reads. A level it has no branch for makes the engine refuse the request rather than fall back, so the scale is a fact about the template. |
| `backend_cls` | Which ullm backend wraps it. |
| `parse_think_tags`, `reasoning_history_field`, `token_limit_param` | The client-side mirrors of the same facts, for a served engine. |
| native `max_model_len` ceiling | A capability, not a choice. What a run uses is P-00. |
| MTP availability, `mamba_cache_mode`, PLE dtype handling | Architecture-specific engine arguments. |
| the input embedding in pinned host memory (`cpu_offload_params={"embed_tokens"}`), Flash-Next | The second-largest table on the card (248,320 rows of 2,560 bf16, 1.18 GiB), read one row a token like the PLE table. The fork's Qwen4Exp method (a public notebook's) keeps it pinned on the host and gathers rows through UVA; the MTP drafter shares the module. On every box the card's 1.18 GiB goes to the KV pool and the host holds the table, so it is the engine's, not a box's. |
| speculative decoding, Flash-Next | Drafting with its MTP head, four tokens a step (the RTX cell's three are P-EV) -- fine-tuned on ARC2 and ARC3 play, NVFP4 experts, a 16k-id draft vocabulary -- sampled and verified against its probabilities, with GDN RecoverSSM and no trailing prefix block dropped. The head's identity is the engine's, as the checkpoint's is; where it is mounted is U-EV. Measured on the B200 replay and the RTX (`packages/ullm/flash_optimization.md`, "Speculative decoding"). |
| conversations kept live, Flash-Next: `prefix_match_unit` 80, `prefix_cache_decode_tail`, `prefix_cache_context_threads` | A cache hit may end every 80 tokens, each decoding request keeps a state at the last unit its decode passed, and the client names each conversation's thread: together what puts the host tier in park mode (U-00). The unit must divide every cache group's block, and the engine refuses one that does not: 80 divides the block at four drafts (1,840) and the undrafted one (1,600). The RTX cell's 64 is P-EV (`packages/ullm/kv_offload.md`, "Park mode"; `packages/ullm/flash_optimization.md`, "Cache hits inside a block"). |
| model-card sampling | The card's own values, overridden by P-00's analyzer profile. |
| `expert_log_dir`, `expert_norms` | The factory's expert routing histogram, off unless a calibration run asks: which experts a checkpoint routes to and how much they contribute, the pruning tool's input. A measurement of the engine, wherever it runs. |

The Flash-Next throughput stack with KN is `ullm.backends.deployment.FLASHNEXT_THROUGHPUT_ENV`, served by the Kaggle RTX Flash-Next cell (the competition kernel) and the traffic arm `kernel_stack`.
Its variables select kernel implementations, not agent policy: sampler folds, compact draft sampling, deferred RecoverSSM commits, strided GDN views, HC fusion, last-row draft prefill, KN's [K, N] layout for the dense GDN qkvz projection, and guarded SM120 CUDA kernels.
The pinned engine owns shape, dtype and architecture fallbacks.
Native play on an RTX PRO 6000 measured 2.84K stock versus 3.36K completed-request tokens/s with it (+18.3%), in one short pair on a uniform205 stand-in for joint205 with an untrained drafter.
The four-L4 cell serves its portable part (`FLASHNEXT_THROUGHPUT_PORTABLE_ENV`): the engine refuses the deferred commits under that cell's pipeline stages, and the SM120 kernels load on no other card (`packages/ullm/flash_optimization_arc2.md`, "The throughput stack on the four L4s").
The B200 cell leaves it off.

## U-0V · task-free, environment only
*Its environment block.*

Facts about the box and the deploy target.

| Setting | What / why |
|---|---|
| `tensor_parallel_size` | Pinned rather than probed: the launcher that builds the args usually has no GPU, and a probe there reads zero. |
| `renderer_num_workers`: 4 on `slurm-b200`, 2 on both Kaggle boxes | The renderer's threads, which process every image of every request before it reaches the engine, with no image cache to skip any (U-00); vLLM's one falls behind a full job's requests. Sized to the CPUs the box spares its games: a cluster job's, fewer on a kernel (2026-10-17; `packages/seal/evidence.md`, "The engine columns"). |
| dense CUDA graph capture, off on `kaggle-4xl4` | Every rung is captured once per card under tensor parallelism, and on those cards capture memory is what set the headroom (`gpu_memory_utilization` there); the box derives none, and its Flash-Next cell names the ladder its traffic replays (U-EV). |
| `CUDA_DEVICE_MAX_CONNECTIONS=32` on `kaggle-4xl4`, for any model | The driver's hardware work queues at their maximum. A pipeline stage's token receive spins on a side stream until the other stage broadcasts, and at the default eight queues a CUDA graph's branch can sit behind it in one, waiting for nothing it needs: ~35 ms a step on the drafted Flash-Next cell, +14.5% generated with 32 (`packages/ullm/flash_optimization_arc2.md`). This box only: no other box here runs pipeline stages, and none has measured it. |
| `hf_hub_offline_after_start`, `HF_HOME`, weights staging | Whether the network is reachable once the engine is up, and where weights come from. |
| which wheelhouse, and how it installs | What a submission installs. The shipped wheelhouse is a Kaggle notebook's output (`arc3-site-0290`) holding the wheels and a record of their install on the pinned image; the kernel replays the record (`ullm.kaggle_site`: 70 s on the RTX against pip's ~190 s) and falls back to pip wherever the record does not match the kernel. |
| the Kaggle image (`KaggleEngine.docker_image`) | Kaggle runs a kernel on the image it last ran on unless told; the engine pins the digest its install record was made on (run H's), through TAAF's `KaggleTarget.docker_image` (R2.65). A new image needs a new record. |
| compile/JIT cache roots | `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, the FlashInfer JIT base. Two vLLM trees reporting one version cannot share a cache key. |
| `TORCHINDUCTOR_COMPILE_THREADS` | Not ours to set: vLLM assigns it 1 at `import vllm` (`env_override.py`, citing its issues #10480 and #10619), over anything set before it and ahead of any `setdefault` after it. The value is always 1, and nothing here writes it. |
| `MAX_JOBS` | The one compile-parallelism knob that DOES reach nvcc, and a memory setting more than a speed one: each parallel job is a `cicc` of several GiB. Sized to the box's cores by `set_env` on the cluster; bounded to four on a kernel, where the model and the host tier already fill the cgroup a compile would run in. |
| cubins vs nvcc | Without either, vLLM silently demotes the MoE backend. With nvcc but no cubins it does something worse than demote: it builds them, and on a box with no network that is the only branch left. The build is one `cicc` of several GiB per parallel job, beside a model already holding most of the cgroup. |
| prebuilt FlashInfer kernels | `flashinfer-cubin` and `flashinfer-jit-cache` in the wheelhouse, pinned to the `flashinfer-python` version because the lookup is keyed by content hash: a drift does not fail, it misses and compiles. `FLASHINFER_NO_DOWNLOAD` beside them, so a miss says so rather than reaching for a network that is not there. A fact about the box -- no network -- rather than about any engine. |
| deploy plumbing | Kernel sources, setup commands, excluded repos, bundle files. Their defaults sit on `RunConfig`; no environment overrides them today. |
| a checkpoint split over datasets (`ullm.kaggle.KaggleCheckpoint`) | A Kaggle dataset counts against its owner's private quota, and expert prunes of one base share every shard the prune leaves alone (Flash-Next's last four hold the PLE table, 42 GiB). So a prune's dataset may carry its index, configs and changed shards and name the datasets that hold the rest. The kernel's first setup command links the parts into one directory (`ullm.kaggle_assemble`), which the engine serves: half a second, and the load reads the same bytes either way. A borrowed shard must hold exactly the tensors the prune's index places in it, at the size its header gives, or the setup fails: a kernel mounts a dataset's latest version, so a re-upload laid out differently would otherwise change the checkpoint silently. A task serving its own checkpoint on a Kaggle cell names the datasets (`deployment.serve_kaggle_checkpoint`); no cell here is split. |
| LoRA adapter identity | Which adapter a server serves. |

## U-EV · task-free, both coordinates
*Its pair block — one branch per cell, ending in a refusal.*

What only the two together decide. Branches rather than a formula: a pair
nobody has served raises instead of borrowing a neighbour's number, because
the failure is silent — the engine starts and profiles a pool sized for
another card.

| Setting | What / why |
|---|---|
| `gpu_memory_utilization`, and one CUDA graph capture | How much of the card this model may take here. Measured on every cell; on a small card what play needs past the start-up profile is an absolute margin. The Flash-Next cells capture CUDA graphs once, after the pool is sized (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, which the fork honours), where vLLM by default captures twice and holds the first capture's estimate back from the pool. Each budget is the old one less that reservation, as the old boot line names it, so the footprint stays where it was measured: 0.97 became 0.9488 on the B200 and 0.9537 on the RTX, and the four L4s' 0.955 became 0.904. Drafting there with its own graph ladder, the four L4s capture FULL decode graphs alone (`FULL_DECODE_ONLY`, a step carrying a prefill running eagerly) and run 0.935: with both graph kinds 0.92 ran out of memory on the second stage after 23 minutes of serving, where 0.91 played the replay window with ~50 MiB of those cards free (each process holds ~1 GiB outside torch), and the PIECEWISE graphs held about half the graph memory for little; with FULL decode graphs alone a ladder from 0.92 to 0.945 put the cell at 0.935 (2026-10-28). The graphs then sit outside the budget (4.33 GiB on the B200, 2.01 on the RTX), and engine start-up fell from 318 to 187 s on the B200 and from 238 to 158 s on the RTX. The 27B cells keep two captures and their budgets. The RTX's 0.975, with two captures, played a replay window but not a nine-hour submission (`packages/ullm/flash_optimization.md`, "The utilization ceiling", "Competitor tricks"; the addendum for the L4s). |
| the allocator (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with `enable_cumem_allocator`), Flash-Next on the RTX | Against the fragmentation that ends a long run; the cumem allocator holds the KV cache because vLLM refuses expandable segments beside a KV connector otherwise. A pair setting: one rank has no custom all-reduce for expandable-segment memory to conflict with, where the four L4s' tensor pairs do (`packages/ullm/flash_optimization_arc2.md`, "Memory in play"). |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 64, Flash-Next on both Kaggle cards | The indexer's prefill scoring buffer grows with the context and the start-up profile reserves none of it; chunked at 64 MiB it stays near the step's other buffers. Measured free on both cards (`packages/ullm/flash_optimization.md`, "Expandable segments and the indexer bound"). |
| checkpoint mount path | Where this environment holds this checkpoint -- and, for a drafting engine, its drafter. A drafting engine's cell must name the drafter: without it vLLM reads the target's own head out of the checkpoint, a second pass over every shard. |
| dataset and model sources (`kaggle_dataset_sources`, `kaggle_model_sources`) | What a submission mounts to get those weights. Kaggle keeps models apart from datasets; no cell mounts a model today. |
| `safetensors_load_strategy="lazy"` behind the kernel's own prefetch (`KaggleEngine.prefetch`), Flash-Next on the RTX | Read the checkpoint into page cache ahead of the loader instead of faulting it in over the mount. The reader (`ullm.kaggle_prefetch`) is the kernel's first setup command, so it reads while the engine installs, a bounded window past the shard the loader has open, dropping passed shards; the engine maps what it cached. The engine's own prefetch reads every shard at once with no bound, from load on, into less RAM than the checkpoint once the PLE table is pinned, and would not choose itself here, since the table is pinned before its free-RAM check runs. The cell names its checkpoints, target first, then drafter: the runner's load order. |
| `safetensors_load_strategy="prefetch"`, Flash-Next on the four L4s | The engine's own prefetch, not the RTX's reader: the reader moves with the loaders it has seen, and four walk the shards here, the second stage's passing the first stage's shards before the first stage starts, so it dropped what those had yet to read (535 s of load against the engine prefetch's 252). |
| `enable_flashinfer_autotune` | Tunes kernel selection for these weights on this card; a 25-game A/B measured it slower on ours, and SM90+ resolves an unset value to True. |
| the MoE backend pin | Which kernels this checkpoint's experts run through here. A pair setting, not an engine one: a backend's support check is a capability family, so each card's cell states its own. The RTX states `marlin`, the fastest backend it admits (`flashinfer_trtllm` is 10.x only and cannot run there); the B200 states `flashinfer_cutedsl` in its W4A16 mode (`CUTEDSL_W4A16_ENV`), the same arithmetic as Marlin's on the B200's own tensor cores (2026-10-15; `packages/ullm/flash_optimization.md`, "The MoE backend"), so the boxes run one arithmetic in two kernels; the decision stays a pair's, made per card. Pinned rather than `auto` in both, because `auto` walks a priority list and logs its rejections at debug, so a demotion reads like a choice. |
| two stages of two tensor ranks (`pipeline_parallel_size` 2, `tensor_parallel_size` 2), Flash-Next on the four L4s | The box's four-way tensor split (U-0V) is the engine's to override: on this checkpoint four ranks pad the 640-wide experts by a fifth and store the attention share of every token twice over 2 KV heads, where two ranks split both (some part of the cache still does not halve over them, untraced). The one PLE layer and its table stay on the first stage, which is the fork's condition. Measured against four stages of one card: the same pool within 1%, ~20% more generation, a stage running half the running set per micro-batch rather than a quarter for two all-reduces a layer within each pair; the price is per-request latency. |
| PLE offload (`offload_ple_table`) | Only one engine has a PLE table, and whether to offload it is about the box's RAM. vLLM's Engram offload (`engram_config.cpu_offload`) creates it in pinned host memory and looks rows up through UVA on a side stream, prefetched a layer ahead; measured at the card's rate on the cluster, where the cell states it. |
| the stage split 27/21 (`VLLM_PP_LAYER_PARTITION`), Flash-Next on the four L4s | The second stage holds the lm head, the sampling and the drafter, so at twenty-four layers each it is the busier stage and the smaller pool. 27/21 is where the pool peaks with the input embedding on the host: the second stage's block is a page larger, layer 26 is GDN so the first stage's block stays as it was, and a twenty-eighth would move attention layer 27 and make the first stage the smaller pool (+3.9% pool, +1.8% generated against 26/22). |
| the drafter, Flash-Next on the four L4s | The engine's drafting, four tokens with the NVFP4 head on the cell's Marlin pin, from its own dataset (`kaggle_dataset_sources`). Drafting three alone added 12% to this box's replay; with the settings tuned around it, 803 tokens a second against 536 undrafted. |
| the CUDA graph ladder (`cudagraph_capture_sizes`), Flash-Next on the four L4s | The graphs the traffic replays: every request count from 12 to 20 a micro-batch, and 28, exact for both the verify rows (five tokens a request) and the draft steps; coarse elsewhere; nothing above 256 tokens, where a batch runs eagerly and its work hides the launches. One list feeds every graph set (PIECEWISE, FULL decode rounded to the verify width, draft steps by request count). ~0.4 GiB a card less than the engine's ladder, and no padding. |

## P-00 · ARC3, neither coordinate
*`DuckAgent` field defaults for what the agent reads; the P-00 block of
`duck.presets.configure()` for what only the engine consumes.*

The duck's shape, and the largest bucket. The split inside it is by reader,
not by kind: a value the agent reasons with is a field, where a sweep reaches
it by name and an override beats it. A value nothing but the engine ever looks
at is stated where the column is composed and written onto the engine args
there — a field for it would be a pass-through, duplicating a setting the
caller can already reach.

| Setting | What / why |
|---|---|
| `context_window_tokens`, `compact_to_tokens`, `safety_margin_tokens`, `per_message_overhead_tokens` | The window contract. |
| `context_window_tokens` (seal) | Longer than the duck's: seal sweep 2's prompt budget, held to the largest window that leaves the summary after a side turn its room under the engine's native window (`_require_summary_fits`; 2026-10-12). It holds more cache a game, so the seal's 27B columns run a share of the duck's slots (P-EV, `max_concurrent`). Task-level, like the rest of the window contract. |
| `engine_ceiling_tokens` | The window the engine was built with, which the P-00 block reads off the engine arguments before the agent settles its own onto them: the checkpoint's native context. What `summarize` runs under. `None` on a hand-built duck. A supplied engine that reports its window settles it to that window when it connects, and cuts a stated one wider than itself, so a summary's cap never asks past the engine actually serving it. |
| `compaction_strategy`, `compaction_ratio`, `summary_thinking`, `summary_tail`, `journal`, `add_files_and_repl_to_summary` | Memory: how a full context is compacted, and the journal beside it. Summarize, with the journal off and the listing on, is what ships (`packages/duck/design.md`, "Summarize"). |
| `chronological_restart` (seal) | The order the history rebuilt around a summary tells its story in. Off, it opens on the summary, then the prompt that started the current turn and the kept call; on, the prompt and the kept call come first, each where it happened, then the summary and the closing message, which were written after both, with a preamble ahead of them all saying the game is under way and where the summary is. Off, a prompt that opened the turn with a level change or a session reset reads as news after the summary, and the closing message then says the session is intact: seal sweep 2's round1 had a model reason through that contradiction. The other wording is the same either way. On from seal sweep 4 (2026-10-26; `packages/seal/evidence.md`, "The defaults of 2026-10-28"). Inert under `prune`. Task-level, like the rest of the conversation's shape. |
| `reply_cap_tokens` | Agent default: 32768. Each request stamps this cap onto its profile copy; the context budget reserves the same number. `None` uncaps generation and reserves 512 tokens, under `prune` only: `summarize` refuses it. Stored profiles are unchanged. |
| `max_tokens_per_level`, `max_tokens_per_game`, `max_moves`, `uncap_when_queue_drained`, the per-game wall net | Budgets. |
| `prompts`, `tool_output_tokens`, `memory_context_tokens`, `python_timeout_s`, `evict_turns`, `max_llm_calls_per_move`, `expose_undo`, `expose_tool_*`, `color_name_legend`, `completion_cutoff_notice`, `multimodal_context`, `multimodal_upscale`, `image_tokens_override`, `frame_api`, `numpy_scalar_repr`, `persistence`, `file_tools`, `wide_sandbox`, `profile`, `inject_text_*`, `prompt_injection_schedule`, `malformed_retries`, `auto_correct_tool_arguments` | The prompt and tool surface. |
| `exact_token_counts`, `token_native_context`, `exact_history` | Accounting and the history promise. Every column serves vLLM, so every column can hold them. |
| `bound_chars_per_token`, `usage_window_tokens` (seal) | Accounting for a backend that cannot count (a hosted API teacher, `tools/teachers/`): the rendered request's characters charged per text token, a tokenizer's calibration in place of the provable bound of 1, and the billed total the window invariant holds responses to, the API's own window where the agent's is a student's. No column sets them, since every column counts by tokens: 1 and the agent's window ship. |
| `compact_before_request_only` | Accounting and the history promise: defer retention trims to the pre-request hook. Default `False`. |
| `require_exact_context` | Accounting and the history promise: require exact counts instead of advisory bounds. Default `False`. |
| `context_observer` | Optional per-game observer. Default `None`; lifecycle contract in `ContextObserver` in [context_observer.py](packages/duck/src/duck/context_observer.py). |
| `color_name_legend` (seal) | Off (2026-10-30): the index legend rides with the first image of each level instead, the seal's boards being indices. The duck's stays on, so `DUCK_LIKE` turns it back on, as `KIMI_STUDENT` does, the Kimi teacher's traces having been recorded with it. |
| `settled_delta` (seal) | Whether a new frame the model did not predict, or predicted wrongly, and a render its test fails on, are described to it as a block of numpy from a frame it knew (`seal.frame_delta`). Task-level: a seal reads it whatever the engine or box. |
| `animation_description` (seal) | Whether a move's animation frames are described the same way: not at all (`none`), on request through `describe_animation` (`available`), or on request and, for a call's longest animation, on their own (`auto`). Task-level, like `settled_delta`. |
| `animation_marks` (seal) | Whether the executed-actions line marks each move that had animation frames, with their count; the first such move in full, with where its frames are and, when animations are described, how to have them described. Task-level, like `settled_delta`. |
| `move_changes` (seal) | Whether the executed-actions report lists a batch's moves one per line, each with how many cells it changed on the board and the colours those cells held before and after (`8 pix changed, colors 5,11->5,11`, or `no change`), ahead of its animation mark in the same brackets. The settled delta explains only a batch's last move. The board's edge is counted with the rest, energy bars included. A level-winning move carries none: its board is the next level's. On from seal sweep 4 (2026-10-26). Task-level, like `animation_marks`. |
| `replay_reminder` (seal) | Whether the prompt says `history` keeps every turn of the level, those before a RESET included, so replaying a sequence shows nothing new, and the turn after a RESET that closed an attempt repeating an earlier one on the level, move for move from its start, says so and that the level is winnable. Replaying and then trying something new is legitimate, so only a RESET closing a repeat fires it; it is not an audit, and it fires every time. On since seal sweep 2 (2026-10-12). |
| `decomposer` (seal) | framediff's search behind every block (`framediff.Decomposer`): its budgets, its effort, its time limit, and its options, `move_distance_cost`, `object_moves` and `animation_patterns`, on (2026-10-08). One step of lookahead for settled frames, plain greedy for animations (`lookahead`, `animation_lookahead`; 2026-10-09). Task-level, like `settled_delta`; a sweep sets its fields directly. |
| `animation_max_chars`, `animation_cut_chars` (seal) | How long an animation's block may run: the limit its stride is chosen to fit, and where a block at a stride the model chose is cut off. Task-level, like `settled_delta`. |
| `preserve_failed_draws` | Retain failed responses and end on them rather than re-sample; requires `malformed_retries=0`. Default `False`; a trainer's switch, contract in `packages/duck/design.md`, "Summarize". |
| `classify_tool_calls` | Agent default `False`. Classify malformed syntax, tool names and argument schemas before observing/executing the response. Enables ullm turn-manager classification and disables episode-cap-lift replay; the backend must use matching classification. |
| `summary_profile` (seal) | The sampling profile every summary samples on, by name in `llm_config.sampling_profiles`: its temperature, `top_p`, `after_reasoning` and `extra_params`, over the live request's seed and injection. Unset, a summary samples on `profile`. Task-level, like the play profile it sits beside; what a sweep sets when it gives the answer its own sampling, since a summary opens past the end of thinking. |
| `malformed_nudges` (seal) | How many times a game is nudged rather than crashed when a play reply is refused on every re-sample (no text and no tool call, or degenerate repetition): the refused draws are dropped and the model told its reply was discarded. Task-level: what a looping reply costs a game, whatever the engine or box. A retained draw cannot be nudged, so `preserve_failed_draws` requires it at 0. |
| `strict_energy`, `pixel_tolerance`, `expose_pixel_tolerance` (seal) | How a world model's renders are held to the frames, the same in `test_world_model()`, the check on every move and the host's test: the starting values of the settings the model changes with `set_strict_energy` and `set_pixel_tolerance`, kept on the host for the game. Strict energy holds a frame to the model's own energy; the pixel tolerance passes a frame the render misses by at most that many cells, under an energy match or not. `expose_pixel_tolerance` is whether `set_pixel_tolerance` is offered at all, and a tolerance only starts above 0 where it is. Off, 0 and on: the setter offered from seal sweep 4 (2026-10-26). Task-level, like the world model they check. |
| `plan_argument` (seal) | Whether the `python` tool takes an optional `plan`, one line on what the call's actions should achieve and what would confirm it, kept on every move the call plays as `history[i].plan`; `"first"` puts it ahead of `code` in the schema and asks for it there, `"second"` after `code`. The schema takes either order, and a call giving only a `plan` is refused for its missing `code`. `"none"`, no `plan`, is the default. Task-level, like the rest of the tool surface: what a model is asked to state while it acts, whatever the engine or box. |
| `expose_undo` (seal) | Whether the game's UNDO is offered, where the game has it: named among the game's actions and played like any move, as the duck's `expose_undo`. Only without a world model (`use_world_model="none"`), refused otherwise: games keep UNDO by different rules, and a world model holding one in its state defeats the search's dedup. Off; `DUCK_LIKE` turns it on, as the duck ships it. Task-level, like the action surface it sits in. |
| `world_model_default` (seal) | Whether, with an optional world model, the system prompt's world-model section and every level's first prompt state attempting one as the default, dropped only once the model is convinced it does not help on the game. Off; `KIMI_STUDENT` turns it on, as the Kimi teacher traces were played. Task-level, like the world model it is about. |
| `diff_image` (seal) | Whether the prompt after a move carries, beside the board image, an image of the cells that move changed: in their new colour, every other cell a navy no game colour uses. It compares the move's frame with the frame framediff described it from, where framediff did; otherwise with the world model's render when a checked move missed it, and with the frame before the move when no world model checked it. None for a winning move unless the level reflection follows (as the settled delta), after a RESET or an UNDO back to an earlier frame of the level (which the settled delta names rather than describes), a move the world model predicted (energy aside), or a move that changed nothing. Only where the board image rides every prompt (`multimodal_context="current_grid"`). Off. Task-level, like the board image it sits beside: what the model is shown of a move, whatever the engine or box. |
| `axis_labels` (seal) | Whether every board image, the audit's boards and the diff image among them, carries row and column labels every 10 cells along its top and left edges, in a slate margin the board sits inside, and the system prompt's image lines say so; the board's own pixels are unchanged. The whale's labels, drawn in the bundled DejaVu (`seal.board_image`). Needs images on and `multimodal_upscale` at least 4; the engine's pixel ceiling grows by the margin. Off. Task-level, like the board image it labels: what the model is shown, whatever the engine or box. |
| `level_reflection`, `reflection_extra_window_tokens` (seal) | Whether a turn follows a move that wins a level, ahead of the next level's first, and how it asks: `verbose`, with three questions, the default from seal sweep 4 (2026-10-26), `concise`, for a brief reflection with none, or `none`. It reports the move as any turn does, then asks the model to reflect on the level and answer three questions, with Python and the file tools and without `submit_action`. Charged to the new level. No summary runs inside it: its context may grow past `context_window_tokens` by the extra tokens, and past that the turn ends and the summary comes with the next level's prompt. The window, the extra, the reply reserve and the summary's room must fit the engine's window. Task-level: what a won level is worth thinking over, whatever the engine or box. The journal refuses it: the turn asks for what the journal's level entry asks. |
| `strategy_audit`, `audit_level_tokens`, `audit_extra_window_tokens` (seal) | Whether a strategy audit runs and how it asks (`verbose`, with two questions, `concise`, for a brief reflection with none, or `none`), and the level-token thresholds it runs at: a turn, not for acting, that asks the model to step back from a level it has not cleared, with advice, the actions it has not tried and two questions, and then play goes on. The spend counts as the level's budget does, less the reflection turn's. Also how far past `context_window_tokens` an audit's context may grow before it ends, as for the reflection. Task-level: when a level has run long is a fact of the play, whatever the engine or box. A mode other than `none` needs thresholds. Concise at one threshold since seal sweep 2 (2026-10-12), which a per-level cap below it never reaches. |
| `suggest_moves_after_tokens` (seal) | How many completion tokens may pass without a move before every tool result of a playing turn ends with a note saying how many have and suggesting experiments over more thinking. Counted on the game's own meter from the last move, summary, or reflection or audit end; a call that moved has just reset it, and a side turn, which cannot move, gets no note. 0 is off. Off until a sweep prices it (2026-10-15). Task-level, like the audit: how long a game thinks without acting is a fact of the play, whatever the engine or box. |
| `response_observer` | Optional callback on each processed ordinary or summary response, before its tools run. Default `None`; contract on `DuckAgent.response_observer`. |
| `max_num_batched_tokens` | The per-step token budget. Run it high: this task trades latency for prefill throughput (the RTX's Flash-Next cell caps it at its largest CUDA graph, P-EV). A constant rather than a field — nothing but the engine reads it. Constrained by U-E0: in `align` mode a prefill chunk must end on a recurrent block boundary. |
| CPU KV tier: on, victim connector | The concurrency is set above what the pool holds, so contexts live in the tier between turns. Its *size* is P-EV, and so would be turning it off where a pool holds the concurrency; no cell does today. |
| the analyzer sampling profile | Swept for this task, and it overrides the model card's. It moves to P-E0 only if several engines run side by side in production. |
| `start_gap_s` (a `RunConfig` default, which P-0V and P-EV raise) | The floor between consecutive game starts. A run's first grants all spawn their child process together, and a couple of hundred spawning at once broke the host; half a second apart is nothing against turns that run minutes, and it is the harness's to pace, not any engine's or box's. |
| `traces`, `capture`, `replay_dir` | What a run records. |
| `trace_content` (`"full"`, no layer overrides it) | What a trace stores: both representations, or the ids alone (TAAF R5.39). A recording made for the traffic replay sets `"ids"` at its launch, not in a column: no deployment records traffic. Only the in-process engine produces ids-only traces; a served backend refuses them at setup. |

## P-E0 · ARC3, engine only
*The P-E0 block of `duck.presets.configure()` — a rule over the engine's own
scale, not a table.*

| Setting | What / why |
|---|---|
| `reasoning_effort` | The swept level for the engine that has one. A level a template has no branch for makes the engine refuse the request, so it is stated per engine rather than as a rule. |
| `zero_recurrent`, `zero_recurrent_at` | Needs a sampling parameter only the fork declares. Supported, currently dormant. |
| `verify_backend` | The backend matrix against the started engine before the first game, for a run that wants its engine proven before it plays; a refused configuration otherwise kills every game on its first turn with the run exiting COMPLETED. Off in every deployment column; the duck-engine GPU columns turn it on for the boot they test, and it carried the Flash-Next bring-up. |
| `compaction_forbidden_token_ids`, the `<tool_call>` marker's id | Banned while a summary is written, so a summary cannot open a tool call. A fact about the checkpoints' tokenizer, which the three engines share, so it is stated once after the engine chain (`duck.presets.QWEN_TOOL_CALL_TOKEN_ID`) and reaches no agent built for another model. |
| `inject_text_*`, emptied for Flash-Next | Its runs so far were played without a served opening, and the register the opening sets halves what this model reasons per action. |
| Flash-Next's checkpoint, the seal's: joint205 (`seal.presets.JOINT205_CHECKPOINT`, on Kaggle `JOINT205_KAGGLE_MODEL` with e307's PLE shards from e307's model) | ARC3's own over ullm's e307: our cut for ARC-3, ahead of e307 on each one's Kaggle box at the deadline (2026-10-28; `packages/seal/evidence.md`, "joint205's columns"). Its drafter, in place of e307's v1 head (2026-10-29), is our combined ARC2/ARC3 MTP head trained with its KV in INT4, the format the seal serves its drafter in (`seal.presets.ARC23_INT4_DRAFTER`, on Kaggle `ARC23_INT4_DRAFTER_KAGGLE_MODEL`): on his replay check against joint205 it led every head at INT4 drafter KV (`arc_3_results/shared/drafter-final-20261029`). The duck plays ullm's. |

## P-0V · ARC3, environment only
*The P-0V block of `duck.presets.configure()`.*

How a run is scheduled where it runs — consequences of the shape of the run
rather than of the model.

| Setting | What / why |
|---|---|
| `slot_policy`, `scheduler_level_weighting`, `slot_hold_s`, `slot_report_interval_s`, `min_request_gap_s`, `min_request_gap_running_fraction` | Who gets a slot when one frees, and how its calls are spaced. Ranking pays only where there are more games than slots and a deadline, which is a submission's shape; Kaggle ranks by each game's whole spend (`least_game_tokens`; 2026-10-12), so the games share the machine evenly rather than cheap winners racing to their end and taking the run under the pool's line sooner. The hold (`slot_hold_s`) is off by default (2026-10-16): a slot changes hands only where a swap is free, at a compaction. The request gap is off by default: under the tier's park mode the tier itself holds a lookup until its conversation's copy lands. Where one is set (a tier running its band wants 5 s) it protects a pool at its admission line, and with fewer than half the slots held it is off, decided per move from the slot tally, so a run's tail is not slowed by pacing that protects nothing. |
| `start_gap_s` 10 s on Kaggle | A cohort started together builds its contexts together; starting games apart flattens the KV peak of that launch wave. |
| the per-game wall net, the per-level cap, `shuffle_games` | The deadline shape. A cap frees a slot by *ending* a game and forfeiting what it would have scored; ranking frees the same slot at no cost, so the cap comes off exactly where the ranking is in play. |
| `play_priority` (empty) | Which games start first, highest first, applied stably after any shuffle; keyed by the id a game will play under (an engine game's env id before it starts). A sweep job sets it to each game's length so its long games do not start late and leave a tail. No column sets it. |
| no vision tower (`ullm.backends.deployment.drop_vision_tower`), for an agent with no image | An agent that sends its frames as text alone leaves every checkpoint's tower as weight the card holds and never reads. Dropped on the submission box only, and only for such an agent: the duck sends the grid as an image by default (P-00), so the tower stays. Decided as the column is built, so an agent made text-only by a later override keeps it. `prepare_for_backend` refuses an image asked of an engine built without it. Environment-only, not a cell: it is true of every engine there. |
| `minimal_diagnostics` / submission mode | Currently a `Benchmark.run()` argument, in no table. |
| `heartbeat_*`, `status_interval_s`, `engine_status_interval_s`, `report_memory`, `label` | Observability. |
| `loop_hold_s`, `loop_hold_every_s` | Off. A test facility: the host's event loop held on purpose, to see what a host stall does to a run, the KV tier's parking above all (`packages/ullm/kv_offload.md`, "Parking on the GPU"). |
| serving topology and its client wiring | In-process engine, in-process HTTP server, or someone else's server. The in-process HTTP server's column sends no image: it continues a history over /v1/completions, which takes none (its factory's `continues_images`), and `prepare_for_backend` refuses an image asked of it. Flash-Next is not served over it: its engine expects the client to name each conversation's thread for the host tier's park mode, that client names none, and the build refuses the pair (`packages/ullm/roadmap.md`, "Park mode over the server topology"). |
| `process_mode`, `subprocess_memory_limit_bytes`, `on_agent_error` | Process hosting. Correctly here, though constant in practice. |
| `checkpointing`, `branch_rng` | Level checkpoints for branched play: `"level"` stores one per (game, pass) as each new level starts, and `branch_rng` says whether a game started from one draws fresh or continues the source's random stream. Which checkpoint a game starts from is the game's own (`CheckpointStart`), not a setting. Set by the run that wants branch points or branches, never by a preset: what a run is *for*, not where it runs. |
| `backend_inactivity_timeout_s`, `accept_malformed_output`, `replay_on_link_loss` | Transport policy: how long a silent request may park a game, and whether a trainer keeps a malformed draw. |
| `engine_stall_timeout_s` = 600 | The run's own watchdog over the engine (`platypus.watchdog`): once the engine has generated nothing for this long while it still reports requests, or has died, the run ends the way the soft deadline does, the reason on the benchmark. How a run ends where it runs, so its home is here; no column overrides it, and `None` turns it off. |
| `rng_seed`, `rng_seed_per_game`, `shuffle_seed` | Reproducibility. |
| `crash_on_framediff_timeout` (seal) | Whether a framediff search that reaches its time limit after a call ends the game, a runaway search being a defect to find, or shows the best block it found by then as any other, the timeout on the span alone. How a run ends where it runs, so its home is here; off on every column, since a lost game costs a sweep more than an undescribed frame (2026-09-29; `SealAgent.crash_on_framediff_timeout` says why). A search inside the model's call stops before the call's limit and never ends the game. |
| `replay_shadow`, `replay_shadow_mismatch` | Whether a RESET, and the actions after it that replay a path already played on the level from its start, are answered from the record instead of sent to the game (`platypus.replay_shadow`): the model sees the same frames, the game is spared actions it would charge to the level and could teach nothing. RESET is taken to restore the board the level opened on, which every catch-up checks. Off: resimulated over seal sweep 2's main1 it moved the sweep score by +0.02, since 65% of what it spares is on levels never won and a won level with long repeats is already far past its baseline, where `(baseline / actions)^2` is flat; and games break its premise (sc25's RESET does not always restore the level; talos_beams' RESET replays an intro a level reached by a win never shows) (2026-10-10). A catch-up the game does not reproduce means the model was shown frames the game would not have produced; on the cluster that ends the game, a determinism defect to find, and on Kaggle the shadow switches off for that game and play continues from the game's real state, since the game's score still counts. |
| `max_attempt_repeats`, `max_actions_per_level` (off) | When a game stuck on a level ends (`platypus.attempts`): one attempt -- the actions from the level's start or a RESET up to the next RESET, `[RESET]` alone included -- played this many times on the level, consecutively or not, each attempt counted on its own; or this many actions on the level, served ones included. Off everywhere (2026-10-10): on seal sweep 2's main1, 100 repeats and 10,000 actions ended 26 of 1,152 games and lost two tn36 games' late wins, -0.01 sweep points at the cut; 10 repeats ended 42 and lost 5, -0.05. The spend they free is mostly what the cut's per-level caps discard anyway, and under the 10-hour game wall neither shortens a job much (77.7 to 76.4 job-hours): they save compute, not score or wall-clock. Task-level, so both agents take it from `RunConfig`. |
| `segmentation` (seal) | Whether the model has `segment()` (with `Segmentation` and `Node`) and the prompt points it there as its first view; off, the kernel lacks all three and the prompt names no starting point in their place. On, as the seal has always run. A switch for a quick test (2026-10-08): if games play worse without segmentation, structure helps, which motivates scene decomposition and framediff work even where today's methods do not. Task-level, like `settled_delta`. |

## P-EV · ARC3, both coordinates
*The P-EV block of `duck.presets.configure()`.*

How much of this model this box can run at once — which neither coordinate
settles alone.

| Setting | What / why |
|---|---|
| `max_concurrent` | Sized to saturate the engine. Flash-Next's is set under the tier's park mode, at or just above what the drafted pool runs, so requests wait for capacity in most minutes without more parked threads than the tier holds: 100 on the B200 and 80 on the RTX (2026-10-12; `packages/ullm/flash_optimization.md`, "What ships"). The seal's, on its own checkpoint, are its own rather than a share of these: 167 and 80 (2026-10-28; `packages/seal/evidence.md`, "joint205's columns"). The 27Bs' sit above what the pool holds, so the host tier is in play: 36 and 18, which are 80 and 40 scaled with the mean context when the prompt budget went from 56,832 to 125,000, times the old budget over the new, so the cache the games hold stays where it was (2026-09-27; `packages/duck/evidence.md`). The seal runs `seal.presets.WINDOW_SLOT_SHARE` of the 27Bs' figures, for its longer window (2026-10-12). |
| `max_num_seqs` | Derived from it, so it arrives here by following. |
| the KV caches, Flash-Next: the sparse-attention cache in INT4, the indexer's compressed-key cache in FP8, the compressor rings in their own pool (`transient_kv_pool`), and the block they take | Chosen on score (2026-10-12); the recurrent state stays bf16, its FP8 rows read out on score. The block is the smallest multiple of the cell's unit that holds the recurrent page at what such a token takes. ARC3's alone until ARC2 adopts them, so here rather than in U-E0, where they then belong. The values are `packages/ullm/flash_optimization.md`'s "What ships", what each buys its "INT4 KV cache" and "The fixed blocks". |
| `max_num_batched_tokens` 512, Flash-Next on the RTX | The largest CUDA graph the card captures, so no step runs eagerly; the engine's activation reserve follows the step budget, so the pool grows, and prefill gets cheaper a token. Not on the B200, where four drafts' decodes alone would fill such a step, so P-00's budget stays (`packages/ullm/flash_optimization.md`, "512 tokens a step"). |
| the CPU tier's size | Claimed eagerly and counted against the same cgroup as the harness, so it takes what that leaves. |
| `prefix_match_unit` 64 and three drafts, Flash-Next on the RTX | The engine layer keys every 80 tokens and drafts four (U-E0). This cell drafts three, since the official games are easier than the private set and lower acceptance erodes four drafts' edge first, and keys every 64, the unit read on this card; its block is chosen as a multiple of the unit (`packages/ullm/flash_optimization.md`, "Speculative decoding", "Cache hits inside a block" and "The fixed blocks"). |
| `start_gap_s`, Flash-Next: 12 s on the B200, 30 s on the RTX | The opening fill of the slots spreads over about 20 minutes, so games reach their compactions apart rather than in one wave (`packages/ullm/flash_optimization.md`, "What ships"). |
| `max_tokens_per_level`, Flash-Next on the cluster | Sized so a game there plays about the tokens a Kaggle submission's does, since the cluster is where settings are tuned and Kaggle where they are scored. The budget itself is P-00; Kaggle lifts it (P-0V), and only this cell resizes it. |

The table is keyed by the pair so an engine states its own rather than
inheriting figures measured for another: the 27Bs share their cells' numbers,
and Flash-Next states the concurrency that keeps requests waiting for
capacity in most minutes, just above what its drafted pool runs, without
more parked threads than the tier holds. On Kaggle, slots past that line
only wait in the tier and dilute the slot ranking.

---

## Where a check goes

The same split decides validation. A rule about **one field** — an unknown
mode, a negative margin — is checked at construction *and* at
`prepare_for_backend`, since post-construction assignment is how settings are
normally applied. A rule **spanning** the agent, its config, the factory or the
engine actually installed is checked only at `prepare_for_backend`: at
construction it would see values that had not arrived yet, and after
construction nothing would run it. `build()` calls that seam once itself, so an
interactive caller still fails immediately.

A **field derived from another field** follows the one-field rule: written at
construction and again at `prepare_for_backend`, idempotently. `__post_init__`
runs once, on the class defaults, before any layer or override has been
assigned — so a derivation that lives there alone is taken from the default
and never again, and an override of the source field lands on the source
alone.

## Judgement calls worth knowing

- **`watermark` is U-00.** A pool kept ~90% full is a general good thing, not
  a property of our workload.
- **Sampling stays P-00**, not per-engine, on the reading that one model is
  the default model rather than one of several kept configured in parallel.
- **`max_model_len` and `max_num_seqs` are derived**, so they live in
  platypus even though they are engine arguments.
- **`gpu_memory_utilization` is set per cell and left conservative.** Tuning
  it against the derived sequence cap is possible later; the loop is
  deliberately not closed.
- **`kv_cache_dtype` is U-E0** as a checkpoint fact: which format an engine's
  cache can take and be trusted in is the model's, not the box's. Flash-Next's
  is the per-token-head FP8 its sparse-attention layers were extended for; the
  27Bs' is per-tensor FP8.
- **A field no deployment sets is still a knob** — for a notebook, a sweep, or
  the ARC2 repo. What ships for it is its default, and that default is placed
  like any other setting.
- **`DUCK_LIKE` and `KIMI_STUDENT` are not layers.** They are P-00 values a
  caller asks for by name at `SealAgent.build`, over whichever coordinates it
  names: playing as the duck, or as the Kimi teacher traces were played, is a
  choice about the agent, not a fact of a checkpoint or a box, since the same
  checkpoint plays either way. A student checkpoint that ships as an engine of
  its own would state them in its P-E0 layer.
