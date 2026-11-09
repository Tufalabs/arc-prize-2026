# Native serving validation

These are historical measurements at the source revisions named below. The
current shared `VLLM`/`EngineProxy` transport is described in
[native_serving.md](native_serving.md); these GPU results do not validate later
transport changes.

The shared Platypus PR74 supports full-game Training PR76 and levels Training
PR77.
Both trainers pin the shared native client through their `platypus` gitlink
and use vLLM 0.25.1. The initial results below used client
`b4d84271ed2501f17f1406c0bd8ba752fca3f5ec`; the newer check records its exact
source versions separately.

## Shared full-suite and proxy follow-up (7 September)

The requested `make test-full` gate passed on shared harness `d45725a`:
1,160 CPU tests, eight Anthropic API tests, two local scheduler tests and all
48 GPU cases across 13 engine columns. The GPU matrix includes real Duck
compaction and continuation. OpenRouter API tests were excluded by request;
16 CPU and three reference-regeneration cases were skipped.

The subsequent shared commit `d0c8b3860e97ed52cdb9dcb8458e71468653b909`
disables uncertain generation replay across native proxy links. Its complete
CPU/type/static gate passed (1,164 CPU tests, 16 skipped), followed by seven
real-engine proxy GPU tests in job 2530764. Socket-loss regressions exercise
one and two proxy hops and prove that a native draw accepted before disconnect
is not generated again. Other backends retain their existing replay defaults.
Both trainers pin this shared commit and use its explicit proxy replay option
instead of carrying separate proxy subclasses.

These checks supplement the frozen full matrix; the unchanged API and unrelated
GPU columns were not repeated on the follow-up commit. They do not constitute
a multi-GPU training or learning-performance result.

## Cache replacement and native parity (7 September, job 2530699)

**Passed:** one B200, scheduler `COMPLETED`, exit `0:0`, elapsed 4m11s.
This check froze clean committed trees:

| Component | Tested commit |
| --- | --- |
| Levels PR77 | `a8419e566507f5c9cc563f3c6f6530b157ae7d68` |
| Full-game PR76 | `5db43cf9e6700b2eb0d91937cf1effbd5c6ebd16` |
| Shared Platypus | `d45725a8c7f330ecd793c228246786455239a517` |
| TAAF source | `774268a104deb36043235972803b8d1c25099261` |

The engine was vLLM 0.25.1 and the model was `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Both native services generated
from the same 4096-token prompt before and after replacing one LoRA alias.
The replacement was a separate adapter copy with 62,390,272 nonzero LoRA B
values zeroed, with distinct source/replacement file hashes.

| Phase in each service | Adapter ID | Cached tokens |
| --- | --- | --- |
| First / repeated draw | 1 | 0 / 3920 |
| First / repeated draw after same-alias replacement | 2 | 0 / 3920 |

Thus the awaited real-engine cache reset succeeded, discarded the old prefix,
and allowed new prefixes to be cached. All 14 subsequent shared `NativeVLLM`
traces exactly matched gateway input/output IDs, sampled logprobs, adapter
identity and tokenizer fingerprint. Six continuation pairs preserved the full
previous input-plus-output prefix. Empty/nonempty reasoning prefill, natural
stop and one-token completion caps all worked.

The PR77 trainer then loaded the replacement adapter and scored 268 sampled
positions from 12 T=1 captures across both services, without an optimizer step:
mean absolute difference **0.0108828**, p95 **0.0831818**, maximum **0.125532**.
These passed the probe's mean<0.05/p95<0.20 gates and the production parity
bounds; no token exceeded the production 2.0 outlier threshold.

This bounded 8192-context test covers native serving, adapter cache mutation,
continuation and sampled-token scoring. It does **not** cover a game rollout,
compaction-summary/reset pair, multi-GPU optimizer cycle or later source edits.
Durable logs, captures, hashes and verification script are under
`/data/personal/$USER/training_results/native_serving_validation_20260907_130506`;
the local verification report is
`work/native_serving_20260907/validation_2530699/report.md`.
The frozen source archive SHA256 is
`c0d7d3b5f7504e8dc4267fd67924728d05d4d45c57b8b0e8ac095dba8243b96c`.

## Initial CPU and integration checks

| Suite | Result |
| --- | --- |
| Levels, exact private harness | 1,133 passed |
| Full-game PR76, exact private harness | 845 passed |
| Shared Platypus `make prepare-ci` | 1,036 passed, 16 skipped; Ruff/Pyright clean |
| Levels, optional harness/vLLM imports disabled | 1,051 passed, 10 skipped |
| Full-game, optional harness/vLLM imports disabled | 823 passed, 8 skipped |

Trainer Ruff, Vulture, shell syntax and dependency-lock checks pass. Public CI
also installs Prometheus explicitly for the native HTTP metrics test; it does
not install vLLM or the private harness.

The harness gates use the exact declared TAAF revision. Tests cover real HTTP
and Unix BackendHost RPC, worker serialization, neutral sampling, admission
before generation, continuation and compaction proofs, adapter swaps and
mismatches, malformed replies, capture/callback linkage and terminal draws.
The levels tests also exercise level completion, summary fallback and replay
memory; the full-game tests preserve journal eviction and fresh game starts.

## Isolated GPU check

**Passed on one B200.** Both branches completed four ordinary/prefilled draws,
two one-token capped draws, and a separate greedy natural-stop draw. All six
continuation prefixes matched exactly; all ten natural-stop responses retained
their sampled EOS ID. LoRA replacement advanced the alias from ID 1 to ID 2,
and every subsequent response kept that exact identity.

| Same-adapter raw logprob comparison | Observed | Gate |
| --- | --- | --- |
| Captures / sampled tokens | 12 / 274 | All sampled tokens aligned |
| Mean absolute difference | 0.0150481 | < 0.05 |
| 95th percentile absolute difference | 0.0980736 | < 0.2 |
| Maximum absolute difference | 0.302132 | Reported; no single-token gate |

The model vocabulary contains 248,320 IDs and the tokenizer contains 248,077,
confirming the 243-token difference handled by the vocabulary fix.
Machine-readable aggregates are in [native_serving_validation.json](native_serving_validation.json).

The check starts the native engine, validates its runtime configuration, loads
and replaces a LoRA alias, and sends real NativeVLLM requests through each
trainer's capture gateway. Initial and continued prompts run both with and
without reasoning prefill. It verifies exact token prefixes, complete sampled
logprobs and the pinned adapter identity. Separate bounded cases exercise a
natural stop and continuation after a one-token completion cap.

After shutting down the rollout engine, the trainer loads the same adapter on
the allocated GPU and scores the captured tokens before any optimizer update.
The parity samples use temperature 1, top-p 1 and disabled top-k; the separate
greedy stop case is excluded from that comparison.

## Bugs caught during validation

- Native engine text is incremental display text: it can omit a terminal stop
  token or differ from decoding an isolated suffix. Parsing now preserves that
  text while training and continuation retain every sampled token ID/logprob.
- Qwen's model vocabulary can contain tokens absent from the tokenizer. Output
  validation now uses the model vocabulary, and input admission matches vLLM's
  maximum model/tokenizer ID rule. Valid model-only tokens remain in the next
  exact prompt; genuinely out-of-range IDs still fail.
- Native client serialization rebuilds transient resources, and the gateway
  retains the lineage metadata required by the native service.

This is a serving and scoring integration check. It does not claim a new
multi-GPU training run or a learning-performance result. Existing training jobs
were left unchanged; game-boundary behavior has CPU harness coverage.
