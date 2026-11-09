ARC-2 training uses a separate pinned `arc2-platypus` submodule and worker
environment. Initialize both submodules before installation:

```sh
git submodule update --init --recursive
uv sync --locked --extra harness --no-editable
uv sync --project environments/arc2-worker --locked --no-editable
```

The independent configurations are `configs/grpo_3gpu_arc2_overfit.json`
(120 evaluation puzzles for both training and evaluation) and
`configs/grpo_3gpu_arc2_heldout.json` (1,000 training puzzles, 120 evaluation
puzzles). Start each in a fresh run directory. Compatible interrupted runs
can resume in their existing directory; checkpoints reject changes to the
environment, algorithm, dataset contents, prompt injection, rollout augmentation
or image-processing stack.

Rollout augmentation is opt-in and off in every shipped recipe but the Flash
one (`configs/flash_8gpu_arc2_grpo_tokmean.json`, both parts on):
`"inference": {"augmentation": {"enabled": true}}`. Each training window draws,
per puzzle, one of the eight dihedral transforms (identity, three rotations,
two flips, two transposes) and a permutation of the colors the puzzle uses
anywhere (train and test, inputs and outputs; other colors stay fixed).
`"dihedral": false` or `"colors": false` turns either part off. The worker
applies both to every grid of the TAAF task before its `ArcGame` holds it
(`augment_arc2_task` in `harness_support/augmentation.py`, called where
`harness_support/arc2/worker.py` builds each slot's game), so all attempts in
the puzzle's group see the same augmented puzzle: its text grids, task and
changed-answer images and kernel arrays. The model answers in augmented space,
and TAAF scores the standing answer against the augmented test outputs; the
reward and the gateway's re-validation read that per-output result. The next
window draws again, and evaluation is never augmented. The draw is a hash of
(`training.seed`, the window's schedule index, puzzle id), so the trainer
re-derives each rollout's augmentation from the record in its metadata
(`augmentation`, also inside `arc2_outcome`) and fails the window on any
difference. The learner trains on the tokens and images the worker produced;
nothing else changes.

A section's `dataset` is one puzzle set or a list of disjoint ones: the packed
`arc2-train` and `arc2-eval`, or `arc2-train-hard` (taaf's derived set of the
47 `arc2-train` puzzles Flash-Next missed on one of two passes, copied into
`harness_support/arc2/dataset.py` because the trainer imports no harness
package). `games` may list the puzzles played from their union; empty plays
all of it. The trainer refuses a listed puzzle outside the union when it
starts, and the checkpoint identity records every set's manifest and the list;
one set played whole keeps that set's manifest, so earlier runs still resume.
The Flash recipe trains on 89 puzzles from `arc2-eval` and `arc2-train-hard`
([training/FLASH_RL.md](../training/FLASH_RL.md#the-recipe-and-smoke-configs)).
Both Flash recipes set `training.checkpoint_keep_last: null` to keep every adapter
and trainer state without pruning or caps: about 0.9 GB per iteration, or 32 GB/day
at 40-minute iterations. An integer >= 2 sets the shared retention count (default 6);
adapters retain their age/pin protections and trainer states always keep the initial
state. The prior trainer state must survive publishing/evaluation before run state
advances. This storage setting may change on resume without changing checkpoint identity.

Full-weight variants are `configs/grpo_8gpu_arc2_full_weight_overfit.json`
and `configs/grpo_8gpu_arc2_full_weight_heldout.json`. They retain the same
puzzles, groups, rewards, sampling, budgets and update settings. GPU 0 collects,
GPUs 1–6 train with FSDP2, and GPU 7 evaluates. Submit the same launcher with
`--gres=gpu:b200:8 --mem-per-gpu=128G` (1 TiB of host memory) and the chosen
full-weight config. Start each experiment in a fresh run directory. LoRA remains
the default for existing recipes.

Full-weight mode trains all language parameters, including embeddings and the
LM head, with FP32 persistent parameters and Adam state and BF16 compute.
The vision encoder and projector remain frozen. Extra training cycles are
supported. Clipping-triggered rollback restores parameters, optimizer state and
step counts across all ranks. Rollout and eval engines receive BF16 weights
over NCCL only between complete collection windows; every episode records the
immutable publication identity. Reference KL must remain zero in full-weight mode.

The full-weight recipes checkpoint every window and retain the latest two
checkpoints. A 27B checkpoint with FP32 parameters and Adam moments is roughly
300 GiB, so budget durable storage and local staging capacity accordingly.
`training.full_weight.checkpoint_every_iterations` and `training.full_weight.checkpoint_keep_last`
control cadence and retention; the first and final iterations always checkpoint.
Run state advances only after the checkpoint is durable. Resume validates the
model, environment, algorithm, dataset and processing identities and restores
optimizer, critic (PPO only), RNG and rollback state. When `SCRATCH_DIR` is set,
the selected checkpoint is copied there once before the distributed restore;
the temporary copy is removed after all ranks finish. This keeps PyTorch's
indexed tensor reads off shared storage.

On B200, full-weight training selects FLA's existing Triton backward backend
before constructing the model. The pinned TileLang 0.1.9 backend failed with a
misaligned-address error for native Qwen attention heads in both padded and
packed GPU probes (job 2571690); Triton produced finite gradients. Vision still
uses FlashAttention. No model, precision or dependency version changed.

Two-B200 job 2571669 exercised a reduced version of the pinned multimodal
architecture with the production FSDP trainer. Uneven token weighting matched
the reference update with zero parameter difference under the existing
`rtol=2e-5, atol=2e-6` check. An extra training cycle ran; a clipped window rolled
back exactly, including optimizer state and step count. DCP restoration was
exact, persistent trainable parameters were FP32, and frozen vision weights
remained unchanged with BF16 computation. Evidence is in
`/data/personal/$USER/arc2-grpo-validation/full-weight-preflight-003/`.
These reduced-model checks do not establish complete 27B training acceptance.

Two-B200 job 2571712 passed native vLLM publication with the full pinned 27B
checkpoint. It streamed 851 BF16 language tensors twice through NCCL, excluding
333 frozen vision tensors and the 15 auxiliary MTP tensors omitted by both
model loaders. Both publications completed; multimodal prompt IDs, media and
generated tokens matched exactly, with zero log-probability difference at
`atol=1e-5`. Unpublished policies and stale proxy identities were rejected.
The sender peaked at 7.11 GiB of GPU allocations. Evidence is in
`/data/personal/$USER/arc2-grpo-validation/full-weight-native-006/evidence-2571712.tar.gz`.
This checks two publications of identical base weights; it does not establish
optimizer execution, trainer/server parity or encoder-byte equality.

Eight-B200 job 2571727 then ran the complete 27B model with six FSDP trainer
ranks, one rollout engine and one evaluation engine. Both 32-episode windows
completed without infrastructure errors, and both full-weight updates passed.
The first trainer/server parity check covered 23,896 generated tokens:
mean absolute log-probability difference 0.01654, p95 0.09479, maximum 0.42572,
and no outliers. The existing thresholds remained unchanged. Sampled training
and evaluation traces had matching media-processing and publication evidence.
Evaluation completed all 120 puzzles with average ARC correctness 0.04167.

The first checkpoint durably saved 851 FP32 language tensors and Adam state
in 24 DCP shards (300.6 GiB). After that checkpoint committed optimizer step 1,
the job received TERM and stopped all trainer and engine processes (expected
exit 143); its scratch directory was removed and diagnostics were archived. The later in-memory
update was deliberately not a resume source.

Resume job 2572045 restored that checkpoint into six fresh trainer ranks and
published it to fresh rollout and evaluation engines. It reused the eight saved
probe states and completed another four groups of eight independent episodes,
without infrastructure errors. Its parity check covered 23,110 tokens with
mean absolute difference 0.01555, p95 0.09038, maximum 0.49938, and no outliers.
The accepted update advanced optimizer step 1 to step 2 at the expected warmup
learning rate of `2.5e-7`. Another complete 120-puzzle evaluation of the restored
policy scored 0.05. The final durable checkpoint committed optimizer step 2 and
`next_iteration=2`, retaining both checkpoints. Slurm and the wrapper completed
with exit code zero after shutdown. This was a bounded acceptance run, not a
sustained overfitting experiment.
Both GPU legs use frozen source `66c547c`; evidence is under
`/data/personal/$USER/arc2-grpo-validation/full-weight-acceptance-001/`.
The subsequent `18919ff` fix preserves application file logging when embedded
vLLM imports its NCCL API; its regression and actual pinned-runtime import
checks passed separately. The `d70475c` checkpoint staging fix passed all 31
focused tests, including real two-rank DCP restoration, in the pinned Kander
runtime. These later fixes did not modify the frozen GPU acceptance source.

Worker construction uses ordinary Duck's `build()` with explicit engine
derivation opt-outs. The trainer's shared `configure_training_agent` composes
its sampling distribution, exact context observer, traces and request pacing.
`EngineProxyFactory` constructs the direct ullm `VLLM`
backend around an engine proxy; it sends engine inputs, not chat requests.
The backend validates whether it can preserve exact history; Duck does not
guess capability from a factory mode. Training supplies the direct backend,
and no chat/server capability bypass is used. Parser and template options
come from ullm's deployment settings. The trainer owns the supplied engine,
so the standalone harness's newer serving pins do not replace the training
stack's pinned vLLM 0.25.1 and Transformers 5.14.1.
The training locks also retain FlashInfer 0.6.13 and its matching cubin
package; they exclude the standalone stack's optional 0.6.18 JIT cache.

Training serves the prompt injection the Duck's engine preset resolves:
`configure_training_agent` leaves `inject_text_on_first_prompt`,
`inject_text_on_later_prompts` and `prompt_injection_schedule` as the harness
set them. At `f380ffd` the 27B preset keeps the class defaults, `"We need solve"`
on a game's first prompt and `"We need"` on every later one (schedule `game`),
and the Flash preset serves nothing. Before 2026-10-10 training overrode these
(the 27B: `"We need solve"` at each context epoch and nothing on later prompts,
schedule `context`), so 27B rollouts now differ from those of earlier runs: every
later prompt opens the reply with `"We need"`, and a summary reset no longer
re-serves the opening. The injected tokens are prompt tokens, never trained on.
The worker reads the three fields off the built agent and every run reports them
(`agent_settings`); the trainer pins the first window's values in the checkpoint
identity (`harness_agent`, `training/rollout_identity.py`) and stops the run when a
later window, or the first window after a resume, reports different ones, naming
the setting. A checkpoint written before this change recorded a
`prompt_injection` the identity no longer holds and does not resume.

The ARC-2 harness pin is PR #41's merge, `f380ffd44b0a31d5bf8e9ddd3d87b9b31729e8f7`;
shared ARC-3 harness/ullm is PR #113's merge,
`e53d822d262c12a3b2b7fb31b52b47f8ce905d67`. Both revisions are pinned together:
the ARC-2 presets require ullm's `KaggleCheckpoint`. Training sets
`malformed_nudges=0` beside `preserve_failed_draws=True`, as the Duck requires:
that nudge discards a reply the backend refused, and training refuses none, so
every malformed draw stays a step available for reward accounting. A retained
empty reply is nudged by the analyzer entry instead, unless
`agent_terminate_on_empty_reply` ends the game on it (below). The worker keeps the deployed
Duck preset, including summary compaction and its token ban, with summary
thinking disabled. The prepared agent's limits and forbidden IDs must match
the trainer contract (below).

**Image pixel ceiling.** The deployed Duck sets its engine's `max_pixels` to
its own largest render (`derive_max_pixels`, `duck.images.largest_image_pixels`),
so the image processor never shrinks an image it sends. The worker turns that
derivation off, because the engine is the server's (`EngineProxyFactory` has no
engine arguments), and the launcher names no `max_pixels`: the served processor
shrinks only above its checkpoint's own ceiling. At window start the worker
reads the server's ceiling from `/metadata` through the gateway
(`image_max_pixels`, null when images are disabled; `docs/native_serving.md`)
and refuses the window when the Duck's largest render, computed by the Duck's
own function for the agent's image scale and gridlines, exceeds it, when the
server has images disabled, or when the server does not state the field
(`check_served_image_ceiling` in `harness_support/arc2/worker.py`;
`tests/test_arc2_image_ceiling.py`).

**Where the trainer can drift from the Duck.** The trainer imports no harness,
so it re-derives the Duck's context limits from the recipe, and loading a
config checks them with its own arithmetic. Line numbers are the Duck's
`agent.py` at `f380ffd`.

| Quantity | Trainer | Duck |
| --- | --- | --- |
| Compaction trigger `max(1024, window - reply - margin)` | `training/context_contract.py` | `_context_budget_tokens` (3492) |
| Reset target under summarize = the trigger | `training/context_contract.py` | `_observer_compaction_target` (3600) |
| Summary cap `2 * (window // 16)` | `validate_config`, `_resized_limits` (`harness_support/arc2/trainer.py`) | `compaction_ratio = 16` (270), `_summary_target_tokens` (2453), `cap=2 * target` in `_summarize_live_history` (2780) |
| Served-ceiling floor `window + reply reserve + 6 * (window // 16)`, reserve 512 when uncapped | `duck_summary_ceiling` (same file) | `_derived_engine_window` (979, the `6 *` at 1011), `reasoning_cap=6 * target` (2781), `_reply_reserve_tokens` (3470) |

The 12288-token overflow headroom above window plus summary cap is training's
own rule, not the Duck's. A Duck change to any of these is caught at window
start, when the worker compares the prepared agent with the trainer's numbers:
the trigger (`context_budget_tokens()`), the reset target
(`compact_target_tokens()`), the served ceiling against the Duck's own bound
(`_derived_engine_window()`, which refuses a ceiling under it), and the summary
cap against the caps the Duck's compaction requests: the worker runs
`_summarize_live_history` on a copy of the prepared agent up to the request it
builds (`_write_summary`), requires its `cap` to equal the trainer's summary cap
and its `reasoning_cap` (the retry with reasoning) to fit under the ceiling
with the window and reply reserve. In the ARC-2 worker environment,
`test_trainer_limits_are_what_the_built_duck_requires`
(`tests/test_arc2_worker_construction.py`) builds the Duck with every ARC-2
recipe's window, reply cap and margin, at the training window and at resized
evaluation windows, and makes the same comparisons, including that
`duck_summary_ceiling` equals window + reply reserve + the Duck's reasoning cap.

The empty reply the journal counts is ullm's and the Duck's: the observer
journals `stop_reason == "malformed" and ullm.backend._is_empty_content(final_content)`,
the Duck's `_last_turn_was_empty_reply` (1944) and ullm's turn manager at
`e53d822d` (`manager.py` 521, which counts it one malformed call and appends
the feedback). It calls ullm's private predicate rather than re-deriving it;
a rename fails the first processed response, and a Duck that changed its own
test would leave termination and the count disagreeing, which only
`tests/test_arc2_empty_reply_duck.py` in the worker environment reports. The
charge does not depend on the flag: an empty reply is kept as malformed, and a
reply kept as malformed counts at least one malformed call. The coefficient
and the cap are the trainer's alone: the Duck neither knows nor bounds them,
and the cap has one copy (`charged_penalty`, `harness_support/arc2/results.py`)
that the worker and the gateway share.

What still passes silently: a recipe the Duck would now refuse loads and
schedules, and fails only when its first window starts (or its first
evaluation window, for a resized evaluation window); a Duck that lowers its
bound or its reasoning cap leaves training stricter than needed, which the
worker accepts and only the environment test reports; and limits training does
not mirror at all: an intermediate summary capped at the reply reserve (1877),
the 5,120 summary requests a game (`_SUMMARY_REQUEST_BACKSTOP`, 88), the trim
and prune internals. The comparisons call private Duck methods
(`_derived_engine_window`, `_summarize_live_history`, `_write_summary`,
`_summary_candidate_source`, `_reply_reserve_tokens`); a rename fails at window
start, as does a `_summarize_live_history` that no longer builds its request
before awaiting anything.

The error policy is the Duck's own since 2026-10-11. The worker sets
`on_agent_error="crash_game"` (`harness_support/arc2/worker.py`), platypus's
`RunConfig` default (`contract.py` 231 at `f380ffd`), which the Duck's presets
keep (`duck_run_config`, `presets.py` 101, sets no other) and the deployment
snapshot records for every column (`tests/references/deployment_snapshot.json`).
Until then the worker set `"raise"`, under which one agent exception ended the
window and the run: on 2026-10-11 the 40-minute Flash run
`arc2-flash-grpo-lt100-aug-40min-20261011` (job 2932277) ended when one game
of 239 hit a kernel housekeeping failure in the Duck's Python tool
(`python_tool.py` 497). A crashed game is scored on its standing answer (below).
The worker states the policy rather than inheriting it, so a harness that
changed its default would not change training; `tests/test_worker_run_config.py`
pins the worker's value. Training's own checks that run inside the agent now
end only their game when they fail: the compaction proofs of
`TrainingContextObserver` (`scripts/serving_contract.py`) and the
generation-identity check of the processed-response journal
(`harness_support/arc2/contract.py`) raise in the game's child, which crashes
that game. The gateway's checks (lineage, sampling, captures, results) still
stop the window.

The 27B ARC-2 configurations use these limits; the Flash recipe plays with the
deployment Duck's ([training/FLASH_RL.md](../training/FLASH_RL.md#the-recipe-and-smoke-configs)):

| Limit | Tokens |
| --- | ---: |
| Agent window | 32768 |
| Ordinary reply cap | 4096 |
| Safety margin | 0 |
| Compaction trigger | 28672 |
| Reset prompt ceiling with a full reply budget | 28672 |
| Summary target / completion cap | 2048 / 4096 |
| Served sequence / training sequence ceiling | 49152 |
| Compaction proof ceiling | 65536 |
| Micro-batch token budget | 40960 |

The served ceiling leaves headroom above the estimated 40k summary sequence,
including an overflow round and tool feedback. `_pack_micro_batches` admits a
sequence above its token budget as a singleton, so that budget stays 40960.
The launcher sizes both collection and evaluation engines from the served
ceiling. Evaluation context overrides resolve through the same helper used by
the collector: the summary cap is `2 * (eval_window // 16)`, the reset ceiling
follows the new prompt budget, and the served ceiling preserves the configured
headroom beyond the agent window plus summary cap. The proof ceiling is
`max(2 * window, served_ceiling)`. Omitted optional summary fields retain cap 0,
an empty forbidden-ID list, and served ceiling equal to the window; ARC-3 retains
those defaults. The absent
`dr_grpo_6gpu_arc2_overfit.json` has no configuration to update.

Gateway admission uses the requested completion budget. An unbiased request that
fits the agent window including its safety margin uses the ordinary reply cap,
even above the full-reply trigger when the episode's remaining budget is shorter.
Over-trigger requests carrying a bias or exceeding that window use the summary
completion cap. Prompt length plus requested completion must fit the served
ceiling. `logit_bias` is forbidden below the trigger, even an empty object. On
summary requests it must equal `{token_id: -100.0}` for exactly the configured
`agent_summary_forbidden_token_ids` (`[248058]` in ARC-2). Integer or decimal-string
IDs and numeric `-100`/`-100.0`
are normalized to integer IDs and floats; missing required IDs, extra IDs,
duplicate normalized IDs, booleans, and different values are refused. An empty
contract permits an absent or empty summary bias. A zero summary cap refuses
every request that needs the summary allowance.

The context observer retains the latest measured candidate and the last served
over-trigger proof. `prepare_request` clears measurements but preserves served
and pending reset evidence; `prompt_measured` supplies the trigger and epoch
for that preparation. A request starts only at epoch 0, continues at the same
epoch with no pending reset, or consumes a pending proof at any greater epoch
with a prompt within the Duck-reported target. For a served summary proof, the
observer expands that target by the unused reply budget when the remaining
episode allowance is shorter than the full reply cap. Stamping clears older
evidence and records only the new request's over-trigger proof, using a matching
current measurement. `history_rewritten(E)` requires the last request epoch and either
a measured candidate or served proof. Only a pending served proof permits
additional consecutive rewrite declarations with no request in between.
Duplicate, out-of-order, undeclared, unchanged-epoch, and stale-proof resets
raise. Measured-proof stamps retain their exact IDs and media; served-proof
reset stamps carry IDs `None` and no compaction media.

The ledger independently requires reset prompts to shrink within its reset
ceiling, adjusted by the same unused reply budget, and the epoch to advance.
Proof is either an unserved over-trigger prompt extending the previous prompt
plus completion, or the previous completed
admitted summary request itself. A continuation must keep the epoch and exact
prefix; any continuation after a summary is refused until a reset. Ordinary
budget-limited replies above the trigger can continue and do not supply served
summary proof. The detached summary therefore continues at epoch E, and the
rebuilt history resets at E+2. The
existing join accepts that epoch advance and produces two training segments.

The backend passes summary bias through `extra_params` into `SamplingParams`,
which EngineProxy serializes into the native request's `sampling_params`.
The native server applies that bias, and the launcher/runtime require
`processed_logprobs` so returned probabilities include it. Capture records the
admitted bias alongside exact IDs and logprobs. Joining, episode packing, and
parity prefixes preserve it only at that draw's completion targets. Collation
allocates one target-aligned bias column per distinct token ID in the batch;
unbiased batches allocate no bias tensors. `_masked_token_scores` adds these
columns to full logits before gathering, normalization, and entropy. Policy,
trust-region, reference, and parity scoring share this path. The parity bands
are unchanged. Ordinary requests retain neutral sampling and unchanged scores.

Processed responses journal `stop_reason`, `has_malformed_error`,
`has_tool_calls`, `malformed_call_count` and `empty_reply`
(`ProcessedResponseObserver`, `harness_support/arc2/contract.py`). Per-step
malformed counts are `max(malformed_call_count, int(stop_reason == "malformed"))`,
so a retained empty summary incurs the same penalty as an empty ordinary reply
without double counting: an empty reply is one malformed call. An empty reply
is a retained reply with no text and no tool call: kept as malformed with
empty final content, by ullm's own `_is_empty_content`, the test the Duck ends
a game on (`_last_turn_was_empty_reply`). A budget stop with nothing in it
keeps its `max_tokens` stop reason and is neither. The worker requires the
`empty_reply` flag on every step and the outcome counts the empty replies
(`empty_reply_count`); the flag is observation and changes no reward, and the
gateway refuses an empty reply reported without its malformed call.
A well-formed tool-call summary keeps count zero; the join records
`RolloutStep.metadata["summary_tool_call"]` for an admitted summary draw whose
processed response contains tools. This is visibility only and adds no penalty.
The summary observer runs before tool refusal, so failed retained summaries
still have response records. An unusable summary ends without retry and scores
the standing submission; the harness reports `gave_up` only when no answer was
submitted, otherwise `won` or `submitted`.

A game whose agent raises ends `crashed` (`on_agent_error="crash_game"`, above)
and the window goes on. Platypus leaves the game unfinished; Benchmark teardown,
when the benchmark ends, marks it `crashed` and scores the answer standing then,
before `benchmark.json` is written, so the worker's one result batch carries it
with its `solver_note` (the traceback). It is scored like `gave_up` and
`cancelled`: the correctness of what it submitted less its malformed-call
penalty, at most the cap, with `termination_category` `"crashed"`; the trainer
requires a terminal state and an outcome that names it (`validate_run_result`).
Its trajectory's status is `CRASHED` (`COMPLETED_RUN` for `won` and
`submitted`, `TIMEOUT` for `gave_up` and `cancelled`; `ERROR` is left for a run
the harness never finished, which fails the run), and its metadata keeps the
note's first and last 300 characters (`crash_note`, `crash_note_tail`: where it
failed and what was raised). Its `run_duration_seconds` runs to the end of the
benchmark, where teardown finished it. A crash before the first model call is a
zero-step outcome. A crash between the host recording a draw and the agent
processing it leaves that last draw unjournaled: the worker drops it, as it
drops a deadline's unprocessed draw, and the gateway admits one unreported
terminal draw for a crashed run. Each window logs `crashed=` in its completion
line and the note's tail for every crashed run; TensorBoard has
`rollout/crashed_count` and `rollout/crashed_fraction`, the iteration record
`crashed_rollouts`. A window in which every run crashed or errored is still
refused. A serving failure is not a crash: the gateway poisons the window on any
failed generation (`tests/test_arc2_crashed_game.py`;
`tests/test_arc2_crashed_game_duck.py` in the worker environment).

Each window contains four uniformly selected puzzles with eight independent
episodes each. Each episode is one full Duck session, including its own two
final attempts. Correctness is the fraction of test outputs for which either
attempt exactly matches every pixel and both dimensions. The terminal reward
is

    reward = correctness - min(max_episode_penalty, malformed_call_penalty * malformed_calls)

with `training.malformed_call_penalty` (default `0.05`) per malformed call: a
malformed tool call, a reply the backend could not parse, or an empty reply
(no text and no tool call), one penalty each. `training.max_episode_penalty`
caps the episode's summed penalty: unset (null, the default) nothing is
capped and the reward is `correctness - malformed_call_penalty *
malformed_calls`, as it always was; set (positive, 0.5 in the Flash recipe),
an episode whose penalty reaches the cap pays the cap instead, so a solved
puzzle with many penalties still scores above an unsolved one. Negative
rewards remain negative, and every penalty is terminal, on the last step. Set
the coefficient to `0` to disable the penalty. The trainer passes the
coefficient and the cap to the worker and independently verifies rewards
against the same configured values (`validate_run_result`,
`harness_support/arc2/trainer.py`, through the worker's own
`charged_penalty`), refusing an outcome scored under another cap. The
outcome keeps the uncapped penalty (`malformed_penalty`), the cap
(`max_episode_penalty`) and whether it bound (`penalty_capped`, the penalty
reached the cap). Checkpoints reject changes to the coefficient or the cap;
older checkpoints retain the historical `0.05` default and, recording no
`max_episode_penalty`, no cap: they resume only uncapped, and a capped
checkpoint only under its own cap. `rollout/avg_malformed_calls` and
`avg_malformed_penalty` (uncapped) count every malformed call, empty replies
included; `rollout/avg_empty_replies_per_rollout` counts the empty ones among
them, for observation; both kinds are `format_break` steps
(`rollout/format_break_fraction`, the collapse watch); and
`rollout/penalty_cap_fraction` is the share of a window's episodes charged the
cap.
GRPO uses the group population standard deviation plus 1e-8. An identical
reward group contributes zero advantage and is not resampled. Generated
tokens retain token-mean loss weighting by default; prompts, prefixes and
tool feedback are masked. `training.loss_aggregation = "episode_mean"` gives
every episode unit mass spread over its own tokens instead.

What follows an empty reply is `inference.agent_terminate_on_empty_reply`.
Either way the reply is a retained, trained step charged as one malformed
call. On, it ends the episode as a give-up, so
its standing answer is scored at once. Off (the Flash recipe since
2026-10-10, as the deployment harness plays), the episode goes on: ullm's turn
manager appends "Malformed reply: no text and no tool call. Reply with a valid
tool call: a tool name and its argument object." to the turn and the Duck's
analyzer entry follows it with its ordinary nudge (`_followup_prompt`, as
after any reply without a tool call), so the next request extends the last
rather than repeating it; with the answer complete the empty reply ends the
entry as a text reply would, and the game stops after its audit turn if that
is still due. `malformed_nudges=0` does not suppress this: that
nudge drops a reply the backend refused and re-prompts, and training refuses
none. Each further empty reply is another malformed call, charged until the
episode's summed penalty reaches `training.max_episode_penalty`, and nothing
caps their number but the game's limits: its token budget (`inference.agent_max_episode_tokens`, none in
the Flash recipe), its runtime and the Duck's 5,120 requests a game under
summarize (`_SUMMARY_REQUEST_BACKSTOP`).
`tests/test_arc2_empty_reply_duck.py` plays the pinned Duck through both
settings with the worker's retention settings (it needs the ARC-2 worker
environment). ARC-3 keeps its critic-based PPO path and charges no reply
penalty: an empty reply is a `format_break` step there, a diagnostic
(`training.format_break_penalty` must be 0), and its config refuses any
`max_episode_penalty` (null is unset).

For GRPO without standard-deviation normalization and with a fixed loss
denominator, use the [Dr. GRPO recipe](dr_grpo.md).

The launcher builds separate immutable worker and serving environments on
node-local storage. Task and `show()` images travel as lossless PNGs, with
ordered pixel hashes, native placeholder layout, processing identity and
processed tensor/position hashes. Training reproduces these inputs without
retokenizing sampled history. Failed rollout images and traces are archived
to the durable run directory. Completed native draws are also journaled on
scratch immediately, so worker termination preserves their token and image
evidence before episode traces are finalized. Correctness (`avg_arc_score`), malformed counts,
the empty replies among them, the uncapped penalty, the share of episodes charged the cap and
shaped reward (`avg_return`) are reported separately.

For a three-GPU Kander run, export a fresh `ARC3_RUN_DIR` on durable storage
and `ARC3_CONFIG=configs/grpo_3gpu_arc2_overfit.json`, then submit
`scripts/platypus_debug_smoke.sbatch` with `--gres=gpu:b200:3` and the appropriate
QoS. GPU 0 collects rollouts, GPU 1 trains, and GPU 2 evaluates. The model stays
Qwen3.8-27B at the pinned revision, BF16, with vLLM 0.25.1 and vision
FlashAttention. Runtime GPU acceptance is recorded separately below as it
completes. `configs/grpo_6gpu_arc2_overfit.json` runs the same recipe on six
B200s with twice the episodes per window: rollout servers on GPUs 0 and 1
collect 64 episodes over eight puzzles (groups of eight, 32 slots per server as
in the three-GPU file), the LoRA trainer on GPU 2 has one data-parallel
gradient replica on GPU 3, and evaluation servers run on GPUs 4 and 5. One
optimizer step consumes `gradient_accumulation_steps` micro-batches per
replica, so that file halves the count to 8 and each step still trains 16
micro-batches; submit it with `--gres=gpu:b200:6`.

CuTe 4.5.2's base and CUDA 13 wheels overlap in their installed paths. The
launcher installs the pinned CUDA 13 wheel last and verifies the actual
experimental module and compiled library hashes before stamping either
environment ready. A new cache recipe prevents reuse of an environment built
without this guarantee. This retains CUDA 13.0.88, FlashAttention 4.0.0b19,
vLLM 0.25.1 and BF16.

Validation evidence: Kander job 2559825 on `dgx02` executed vLLM's actual BF16
vision FlashAttention 4 wrapper with unequal image sequence lengths. Its
maximum absolute difference from the FP32 reference was 0.003785, below the
unchanged 0.02 threshold. The complete Qwen3.5 vision architecture executed
all 27 noncausal FlashAttention blocks with finite output. Both serving
processes also passed image-encoder profiling with the pinned checkpoint.
The structured probe and log are in
`/data/personal/$USER/arc2-grpo-validation/vision-preflight-001/`.
The same directory contains the isolated reproduction of the base wheel's
unconditional unsupported-toolkit exception. The zero-update job completed
with exit status 0; the production environment was never modified to inject
the failing package.

CPU checks against the real pinned Transformers processor and Qwen3.5
position computation matched exactly for task images, appended images,
repeated images, pruning, packed sequences and mixed padded batches.
Actual native rollout records also verified a `show()` image append and
pruning both with and without retained images, including exact sampled
history, image order and processing evidence. The selected records are in
`/data/personal/$USER/arc2-grpo-validation/overfit-003/media-lifecycle-acceptance.tar.gz`.
Kander job 2561129 then exercised the complete pinned policy on a small mixed
image/text batch, packing three samples into two sequences. Generated-token
logprob differences between packed and unpacked forwards met the existing
thresholds (mean 0.0352, p95 0.1563). Backward produced finite, nonzero
language-LoRA gradients with no vision gradients and no critic. This probe
took no optimizer steps. Its report and image evidence are in
`/data/personal/$USER/arc2-grpo-validation/trainer-mm-diagnostic-003/`.
The complete three-B200 run in
`/data/personal/$USER/arc2-grpo-validation/overfit-006/` passed collection,
optimization, evaluation and cancellation/resume acceptance. Job 2567937
collected all 32 episodes in four complete groups of eight, including an
identical-reward group with zero advantage. Actual rollout-to-trainer logprob
parity passed over 20,070 generated tokens (mean absolute difference 0.01732,
p95 0.09837). The first accepted update had mean ratio 0.999964, clip fraction
0.000193 and gradient norm 0.02311. Evaluation completed all 120 puzzles with
zero infrastructure errors and equal-puzzle correctness 1/120; malformed
counts, penalties and negative training rewards were preserved separately.

Normal full-job cancellation left no failure record or surviving job
processes and preserved the accepted checkpoint. Job 2568192 resumed the same
frozen source, restored all 800 Adam states and regenerated a byte-identical
runtime adapter on its own scratch storage. The resumed all-wrong window was
retained without resampling; its malformed-call penalties supplied the reward
variation. Parity passed over 22,151 tokens (mean 0.003950, p95 0.01972).
The next accepted optimizer step continued warmup at 2.5e-7, with mean ratio
0.999918, clip fraction 0.000259 and gradient norm 0.02111. Both checkpoints
have matching ARC-2/GRPO, dataset and processing identities and no critic state.
The resumed evaluation completed another 120-puzzle cycle with correctness
2/120 and zero infrastructure errors. These are bounded execution and early
stability checks; they do not establish sustained overfitting or a statistically
reliable learning improvement. The generic format warning in the first window
described the base policy's malformed responses before its first update.

Structured evidence is stored in `acceptance_first_window_*.json`,
`acceptance_resume_*.json` and `acceptance_evidence/` under that run directory;
the corresponding rollout archives preserve images and exact draw histories.
The acceptance run uses source revision `b2e4440`, ARC-2 harness `611ec548`
and shared harness/ullm `1e452040`. Later documentation-only commits do not
change its frozen source.

Training rejects repairable malformed tool arguments before executing them,
preserves the sampled text, and appends explicit feedback and the penalty.
This behavior was confirmed by the owner on September 11, 2026. Standalone
harness behavior retains its existing repairs.

The PR #25 integration uses the upstream training hook unchanged, with the
context opener enabled by the worker. Kander compatibility job 2568663
passed all 1,171 trainer tests, 52 focused shared-backend tests and 66
ARC-2 harness tests in the isolated training/worker environments. The
trainer and worker retained vLLM 0.25.1 and Transformers 5.14.1. This
checks the combined source; the GPU acceptance evidence above belongs to
its explicitly listed earlier source revisions.

After its second accepted update and completed evaluation cycle, the
earlier resumed job 2568192 stopped on an exact-history continuation guard
in a later window. The fatal stop and diagnostics are preserved in
`overfit-006`; the two completed acceptance windows do not establish that
longer training is free of continuation failures.

Shared harness CI at `e475b711` passed formatting, lint, strict typing and
1,435 cheap tests (16 skipped). The ARC-2 pin `25970e3` includes PR #25 head
`c678b4a`; its CI passed formatting, lint, strict typing and 1,189 cheap tests
(4 skipped). Trainer CI at `b72416a` passed its checks and 1,047 tests
(99 skipped). These CI checks complement the Linux compatibility run above,
which exercised the actual vLLM-dependent trainer and isolated worker tests.

PR #25 GPU acceptance job 2569193 loaded both engines and passed vision
profiling, then stopped before an optimizer update because the gateway's
default HTTP pool admitted only 100 of the 120 evaluation requests. Queued
requests exhausted the connection timeout. A loopback regression reproduced
that exact admission ceiling; the gateway now follows EngineProxy's unlimited
connection pool, with concurrency bounded by the configured rollout slots.
Generation remains single-attempt. The failed run's archived diagnostics
remain in `pr25-acceptance-001` on durable Kander storage.

The next PR #25 acceptance job, 2569489, admitted all 120 evaluation
requests and verified opener placement across 488 sampled responses and
three context epochs, but stopped on the exact-history continuation guard
before completing its first training window. Its archived diagnostics are
in `pr25-acceptance-002`. A subsequent CPU stress test against the actual
vLLM 0.25.1 Qwen3 reasoning parser reproduced shared mutable parser state:
1,416 of 2,000 overlapping calls produced incorrect splits; serializing
parser access produced zero mismatches. Shared ullm `1d47875b`, also pinned
by ARC-2 `ceb9ded`, serializes parser access across history rendering,
final responses and partial responses.
Both added regressions fail against the original backend, and Kander CPU
job 2570737 passed all 110 focused tests with the fix. The exact-history
guards, no-retry policy and pinned engine remain unchanged. This regression
evidence does not establish GPU acceptance. Bounded three-B200 job 2570770
uses trainer `a759384`, ARC-2 `ceb9ded` and shared ullm `1d47875b`, with
durable diagnostics in `pr25-acceptance-003`. It completed both windows and
two accepted optimizer steps, with 32 episodes in four complete groups per
window. The first window's archived
results verify 32 distinct seeds, exact token/draw accounting, analytical
GRPO advantages and mean reward 0.28125 (correctness 0.375 minus 0.05 times
1.875 malformed calls). Its 21,936-token parity check passed: mean absolute
error 0.0163371, p95 0.0938443, and zero outliers under the unchanged
thresholds. The checkpoint contains finite Adam state for 800 parameter tensors
and no critic. One full 120-puzzle evaluation completed with correctness
4/120 and zero infrastructure errors. Slurm and the wrapper exited 0.
Normal shutdown exposed a launcher ordering issue: signaling the whole
engine process group interrupted pending requests before the HTTP host
could drain them. The launcher now signals the host first, retaining the
30-second group-kill fallback. A real parent/child-process regression fails
on the prior behavior and passes with that change. The shared follow-up
`e475b711` changes only GitHub runner disk/cache setup and has identical
runtime source.

The final ARC-2 merge `25970e3` includes PR #25 head `c678b4a`: backend-owned
capability checks and the agent's authoritative reply reserve. The temporary
local reply-cap ordering workaround is removed; `prepare_for_backend` and
the configuration helper used the pinned upstream implementations in that run.
Kander CPU job 2571286 passed 30 focused tests and compared the settled training/evaluation
configuration and system prompt against the successful GPU source: both
are byte-identical. The regenerated lock pins shared ullm `e475b711`.

Targeted shutdown job 2571502 passed on one B200 with the production
launcher at `bd304ee`, vLLM 0.25.1, BF16 and vision FlashAttention enabled.
All nine launcher checks passed. After a warm-up request, the launcher
received a stop while a 512-token native draw was pending. That draw
completed normally, retaining its exact prompt IDs; the host exited 0 in
10.01 seconds, without a fatal engine error, traceback or leaked-semaphore
warning. Slurm exited 0 and released the allocation. The diagnostic archive
is `/data/personal/$USER/arc2-grpo-validation/pr25-shutdown-gpu/diagnostics-2571502.tar.gz`.
The preceding diagnostic attempt failed before generation because its
standalone tokenizer invocation omitted Transformers 5's `return_dict=False`;
the production backend already supplies that option. No serving guard or
production token handling changed to make this probe pass.
