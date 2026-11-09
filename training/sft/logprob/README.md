# Mixed Kimi CE + KL cosine SFT

The frozen corpus contains 1,934 training rows and 35,643,612 supervised tokens per pass,
with 17,435,232 KL-eligible positions (48.92% coverage). CE applies to every supervised token.
The two-pass run has 968 updates; cosine reaches its floor at zero-based update 483.

The reviewed recipe is [`reap307-kimi-mixed-cosine.json`](../../configs/flash_sft/reap307-kimi-mixed-cosine.json).
Its data comes only from the 19 collections and archive SHA256s in
[`reap307-kimi-mixed-sources.json`](../../configs/flash_sft/reap307-kimi-mixed-sources.json).
Preparation never discovers new collections. The mandatory-world-model collection is explicitly
excluded because its teacher prompt differs from the `world_model_default` student baseline.

Against the October 29 native CE baseline (#118), the three requested changes are:

1. CE on every supervised completion token, plus 0.3 recorded-teacher KL at usable positions.
   Completions without logprobs receive CE. Flash keeps its tokenizer.
2. PR #122's cosine learning-rate schedule: five warmup updates, decay from 1e-4 to 1e-5
   over the first pass, then a second pass at 1e-5. The boundary is frozen in the config.
3. The expanded, explicitly frozen Kimi ARC2 and ARC3 corpus, without reward filtering.

Additional deliberate differences are reasoning-enabled summaries from the current
`kimi_student` profile, checkpoints every 20 updates, and the official PR #131 evaluation
protocol with explicit FP8 KV. Summary rendering is selected from `agent.summary_thinking`
and passed to the converter as `--summary-thinking` or `--no-summary-thinking`.

The native trainer, frozen NVFP4 experts and routers, attention/shared/routed LoRA ranks
128/256/16, host AdamW, seed, batch of four rows, and 192k sequence cap retain the native
baseline settings. Four B200s preserve that global batch; the experiment cap is eight.
All five agreed ARC3 holdouts and Crane Dipper are excluded from training. Sliders is
also excluded to match our baseline, independently of the holdout policy.

## Alignment scope (`sft/logprob/targets.py`)

Flash is trained on the same semantic completion as Kimi produced, rendered
through Flash's own chat template, exactly as the CE objective already is. The
two templates differ only in their wrappers, so the KL is attached wherever the
generated text is the same under a semantically identical prefix:

- **Reasoning**: Kimi's `reasoning_content` tokens, found verbatim inside
  Flash's `<think>` block. The converter strips the reasoning at its edges;
  teacher tokens not entirely inside the surviving bytes stay CE-only
  (`trimmed_teacher_tokens`, never split or re-encoded), whitespace-only
  reasoning counts as none (`whitespace_only_reasoning_steps`), and the
  end-of-reasoning decision is only attached when nothing trailing was trimmed.
- **Content**: Kimi closes reasoning with `</think>` and writes the answer
  inside `<response>…</response>` (a final answer ends inside it); Flash renders
  the trimmed content after `\n</think>\n\n`. Leading/trailing whitespace Kimi
  emitted is trimmed by Flash's template and stays CE-only.
- **Tool calls**: Kimi K3.1 emits `<tools><call tool="NAME" index="N"><argument
  key="KEY" type="TYPE">VALUE</argument>…</call></tools>` with control tokens as
  brackets and raw string values; Flash renders `<tool_call>\n<function=NAME>\n
  <parameter=KEY>\nVALUE\n</parameter>\n…</function>\n</tool_call>`. The tool
  name, each parameter key and each argument value are aligned; Kimi's `index`
  and `type` attributes and both wrappers are never aligned. String values are
  byte-identical on both sides. Non-string values (arrays, booleans) are only
  aligned when Kimi's JSON spacing equals Flash's `tojson` rendering, otherwise
  they stay CE-only (counted as `argument_values_unaligned`).
- **Residual bucket**: a Kimi alternative that is a control token (e.g. "close
  the argument here") has no bytes and stays in the residual bucket, compared
  with the student's mass outside the mapped IDs, which is where Flash's
  `\n</parameter>` lives. Kimi's sampled control tokens are never events, so
  when to end a value or a call is learned by CE only.
- **Compaction summaries** stay excluded: Flash is served the summary without
  reasoning while Kimi reasoned first (five training steps).

Every Flash literal is verified while walking the rendering, and the parsed
markup is cross-checked against `output_message.tool_calls` (call count, names,
argument key order, non-string values); any disagreement raises. Coverage
counters distinguish `reasoning_`, `content_`, `tool_name_`, `parameter_key_`
and `argument_value_` positions and teacher tokens; `markup_teacher_tokens`
counts the control and tag tokens that are never aligned. Surveyed over all
9,923 rearc-rest and arc3-hard-r2 steps: 11,690/11,690 string argument values
equal their `tool_calls` JSON value byte for byte, 62/118 non-string values
differ only in spacing, and no step deviates from the grammar above.

Boundaries need not be one-to-one. A teacher token that is longer than the
Flash token starting at the same byte (Kimi's `26` or `.\n\n` where Flash
spells `2`,`6` or `.`,`\n\n`) is attached to that Flash token: its probability
goes to the token Flash actually produced, and the Flash tokens that spell the
rest of its bytes stay CE-only (`prefix_positions`). A teacher alternative that
is no Flash token (Kimi's `123` where Flash has `1`) is projected onto the
first token of Flash's own BPE spelling of its bytes, computed from the
tokenizer's merges (`projected_alternatives`, `projected_mass`), so its mass
counts for `1` instead of the residual bucket; several alternatives can land on
one Flash token and add up. Partial UTF-8 fragments are never projected.
`prefix_positions_projection_differs` counts how often that standalone
spelling picks another first token than Flash chose in context. In the reverse
case, teacher tokens that together spell one Flash token (Kimi's `/button` +
`s` where Flash has `/buttons`) are merged (`merged_positions`,
`merged_teacher_tokens`): the Flash token gets the product of the sampled
probabilities along the path, and every alternative observed along the path
carries the product of the sampled probabilities before it times its own, as
the bytes it would have spelled from the Flash token's start, projected like
any other alternative. A path that crosses the Flash token's end, or runs into
a control token (Kimi's `()` then the close-argument token where Flash has
`()\n`), stays unaligned (`*_teacher_tokens_shorter_than_student`,
`*_teacher_tokens_no_boundary`).

Kimi's control tokens carry its act-or-continue decisions: stop reasoning,
stop the content, end a value, add another argument or close the call, make
another call or end the calls. They have no bytes, but each begins a branch
that Flash's template also begins with a definite token: `\n` before
`</think>`, `\n\n` before `<tool_call>` or `<|im_end|>` after the content,
`\n` before `</parameter>`, `<` or `</` after a parameter, `\n` or
`<|im_end|>` after `</tool_call>`. The sampled control token's probability is
attached to the Flash token at that position (`decision_positions`,
`{kind}_positions` for `think_close`, `content_close`, `value_close`,
`argument_or_close`, `call_or_end`); other control tokens among its
alternatives are the other branch, whose identity follows from the binary
grammar; text alternatives are projected as anywhere else. A control token
among the alternatives at a text position means "stop here" and is attached
to the token that begins the region's wrapper (`special_alternatives`,
`special_alternative_mass`). The decision after the last call of a step and
after a final answer is not in the stream (no end-of-turn entry is logged), and
one whose Flash position does not begin the wrapper, or that Kimi made after
whitespace Flash trimmed, is counted in `decisions_unplaced`.

`scripts/flash_sft_logprobs_20261029/targets.sbatch` rebuilds the four row
files of the running experiment with this scope (row order and baseline fields
preserved), staging the step files to node-local scratch. Job 3633834 did so
as a measurement; its row files were retired on 2026-10-30 because those rows
render summary turns without reasoning (the student now reasons on summaries,
and PR 121's `scripts/flash_mixed_cosine120/prepare.py` re-renders from the raw
traces and runs this module on the result). Only the coverage reports remain,
under `/data/shared/research/arc_3_results/shared/reap307-kimi-logprob-toolcalls-20261030/reports/`.
Every reasoning position of the original targets was reproduced with identical
ids; the additions were:

| | completion tokens | KL positions (reasoning only) | KL positions (this scope) |
|---|---:|---:|---:|
| train (387 rows) | 6,002,711 | 3,797,742 (63.3%) | 4,838,661 (80.6%) |
| validation (8 rows) | 96,724 | 56,724 (58.6%) | 78,950 (81.6%) |

In training, 812,131 of 918,513 argument-value teacher tokens and 208,715 of
225,213 content tokens align; 59 of 11,416 argument values (non-string JSON
spacing) and the 5 summary steps stay CE-only. The wrapper, control and tag
tokens (401,065) are never aligned.

With longer teacher tokens attached and alternatives projected (job 3634559,
same rows, reports under `measure-prefix/reports/`), training reaches
5,254,608 KL positions (87.5%): 415,947 prefix positions (330,316 reasoning,
73,428 argument values, 12,203 content), 408,007 projected alternatives
carrying 53,404 of the 5,243,781 mapped mass, and the standalone spelling
differs from Flash's context choice at 6,035 prefix positions (1.5%). Every
remaining unaligned teacher token is a teacher token shorter than the Flash
token at its boundary or one sitting inside such a token: 62,638 reasoning,
32,659 argument-value and 4,295 content tokens. Validation reaches 84,831
positions (87.7%). Merging the teacher tokens that spell one Flash token (job
3635030) adds 38,865 positions from 77,763 teacher tokens: 5,293,473 (88.2%)
in training, 85,499 (88.4%) in validation, leaving 12,488 reasoning, 7,328
argument-value and 2,013 content teacher tokens unaligned, all on paths that
cross a Flash token's end or run into a control token. Attaching the control-
token decisions (job 3635129) adds 49,164 positions in training: 9,121 stop
reasoning, 8,916 stop content, 10,495 end value, 20,482 add argument or close
(every one), 150 another call; 921 decisions were unplaced. Control-token
alternatives at text positions moved 1,067 of the 5,332,766 mapped mass out of
the residual bucket (6,004 entries). Training reaches 5,342,637 positions
(89.0%), validation 86,443 (89.4%).
No config points at retired data. A launch on re-rendered data needs its own
config and `preflight.py` expected counts, since the script asserts the
reasoning-only numbers.

## Tool-call-aligned experiments (`scripts/flash_logprob_toolcalls_20261030/`)

Three arms re-target PR 121's frozen rows (1,934 training rows, the same 20 validation-probe
rows, every baseline field byte-identical) with the alignment scope above: 24,103,988 of
35,643,612 supervised tokens per pass carry KL (67.6%; the baseline's 17,435,232 is 48.9%).
The 8,760,831 tokens of completions recorded without logprobs stay CE-only in every arm.
Each recipe differs from `reap307-kimi-mixed-cosine.json` only in the data files, the W&B
identity and `recorded_logprobs`: `reap307-kimi-mixed-toolcalls` keeps CE on every token plus
0.3 KL; `-xcl03` and `-xcl10` train a token with a recorded target by KL alone (0.3 or 1.0)
and every other supervised token by CE alone (`ce_positions: "unaligned"`).
`tests/test_flash_sft_toolcalls_recipes.py` and the launcher enforce that rule.

## Loss and alignment

For N supervised tokens and eligible positions A:

`loss = (CE_sum_over_all_tokens + 0.3 * KL_sum_over_A) / N`.

Original reasoning, content and tool-call bytes receive KL where the alignment scope
above places them; eligibility can recover at later boundaries after a token split or
merge. The converter strips reasoning at its edges; alignment accounts for those removed
bytes and leaves partially trimmed teacher tokens on CE without splitting or re-encoding
teacher tokens. The remaining template wrappers, absent logprobs, and unmatched positions
receive CE.

Mapped top-k alternatives retain their original API probability mass. Unmapped and
unreported mass stays in an explicit residual bucket, compared with the student's mass
outside the mapped IDs. There is no top-k renormalization or reconstruction of full
teacher logits. This is a coarsened cross-tokenizer objective, distinct from ALM chunk matching.
The coefficient 0.3 is an initial experiment setting, not a tuned value from the ALM paper.

The data audit reports completion tokens, eligible positions, coverage and per-source
counts. No coverage figure is assumed before the frozen data finishes building. Every
validation pass reports `ce_loss`, `teacher_kl`, `ce_plus_kl_loss` and `kl_coverage` on the
same held-out rows and token denominator, globally and by game/source. Corrupt recorded
bytes, conflicting probabilities, invalid targets and nonfinite losses fail the run.

## Reproduce and qualify

Shared source, harness, runtime capsules, checksums, pytest dependencies and launch scripts are staged at:

`/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/artifacts`.

`stage_checks.py` verifies capsule hashes before use. Bootstrap records package-tree
hashes; preparation verifies that the imported packages match those trees and records
the resulting harness identity in `RENDER_CONTRACT.json`. A literal commit string alone
is insufficient provenance.

```bash
python -m pytest tests/test_flash_sft.py tests/test_flash_sft_logprobs.py \
  tests/test_flash_sft_logprobs_sharded.py tests/test_flash_sft_cosine120.py \
  tests/test_kimi_conversion_diagnostics.py tests/test_mixed_sft_provenance.py tests/logprob -q

TASK_ARTIFACTS=/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/artifacts
sbatch "$TASK_ARTIFACTS/prepare.sbatch"
# After the preparation job succeeds, submit the gated training launcher:
sbatch --dependency=afterok:PREPARATION_JOB_ID "$TASK_ARTIFACTS/train.sbatch"
```

Preparation must reproduce the committed data hashes and schedule before publishing
`DATA_READY.json`. For an intentional recipe change, `--build-only` produces a candidate
without authorizing GPU training; freeze its exact config in Git and use `--verify-only`
to compare and publish it. An unfrozen or changed candidate fails closed.

Preparation cache keys include input archive and benchmark hashes, converter source,
rendering files and harness package hashes, and the actual conversion flags. Packed-row
keys include the converted receipts, packer source and packing flags. Legacy or changed
keys trigger rebuilding; the stage receipts and data audit record this provenance.
This cache hardening applies to future preparation and does not change the already
frozen corpus or the running experiment's source capsule.

The training launcher reruns CPU tests and model staging, then executes a real isolated
four-GPU update containing both CE-only and CE+KL rows. It requires finite loss, correct
CE/KL accounting, unchanged routers, updated/synchronized adapter groups and a complete
checkpoint. Production starts with fresh adapters only after qualification passes.
Bounded production segments checkpoint and resume optimizer, RNG and adapters exactly.

The earlier logprob-only experiment is historical; its row hashes and collection audit
are recorded in the PR description, not used by this recipe.
