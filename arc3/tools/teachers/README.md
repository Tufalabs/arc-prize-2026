# Distillation teachers

*Register: living.*

Kimi K3.1 Pro played through Moonshot's API, to record TAAF traces a
Qwen3.8 Flash-Next student is SFT-trained on (one student for ARC2 and
ARC3; ARC2's teacher runs live in arc2-platypus's `tools/teachers/`).

| File | What |
|---|---|
| `teachers.py` | The teacher entry: model, key variable, the measured factors behind its accounting, its slots |
| `seal_smoke.py` | The launcher: the seal on the teacher's API over a game set, as a CPU job; `--traces` records |
| `run_matrix.py` | ullm's live backend matrix against the teacher (paid, minutes) |
| `image_stress.py` | Whether the API keeps a conversation's images apart and in its cache (paid, minutes) |

The key is `KIMI_API_KEY` in the launching shell's environment. A launch is
`uv run python tools/teachers/seal_smoke.py --api kimi-k3.1-pro --set main
--traces`; the module docstring lists every flag. Runs go under
`/data/shared/research/arc_3_results/<user>/distill/kimi-k3.1-pro/`.

## The student

A model trained on these traces plays the seal they were recorded on:

    SealAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200", kimi_student=True)

`kimi_student` sets `seal.agent.KIMI_STUDENT`, the model-visible settings
the teacher played with that the column has off: the world-model default in
the prompts, and summaries written after the model's own reasoning. The
teacher's launcher builds from the same dict. What else the teacher sets is
for its API and does not carry: the estimate-mode accounting and its
budgets in the teacher's tokens, no banned token in a summary, no host
watchdog. The reasoning level stays the column's `xhigh`, the level the
traces are rendered at in training. Sampling stays the column's too;
whether a student should sample as the teacher was sampled (temperature 1,
top_p 0.95) is open (`packages/seal/roadmap.md`).

The checkpoint is the engine coordinate's. The B200 column serves joint205
in NVFP4 beside the combined ARC2/ARC3 drafter head: its MoE backend needs
NVFP4 weights, and a drafter is only valid for the checkpoints it was
trained against (`packages/seal/roadmap.md` has what is open). A trained
student's checkpoint goes in with `SealAgent.build(..., checkpoint=…)` on the
B200 column, in joint205's place beside the same drafter;


## The traces

`--traces` records at TAAF's hosted-API grade (`taaf.trace.v2`), one trace
per game, bundled into the job dir at teardown:
`retained_traces/trace_bundle.tar.zst`, with `trace_manifest.json` mapping
each run id to its member prefix. A trace is `trace.json`, `steps.jsonl` and
`media/`. Each step is one model call: the whole message list it was sent
(past reasoning included), `output_message` with the reasoning and the tool
calls (`output_text` is the visible reply alone), and the action it played
where it played one. A game stopped early (`cancelled`) keeps every step up
to the stop.

`extra["teacher_logprobs"]` holds the API's per-token alternatives as
returned: a list over the output tokens, reasoning included, of `token`,
`bytes`, `logprob` and the top 20 alternatives in the same shape. The API
leaves its section markers out, about ten tokens a call, and gives real
values only for its few likeliest alternatives: the rest read `-9999` (86%
of the alternatives in a pathfinder trace). The tokens are Kimi's, not
Flash-Next's.

### Against Flash-Next's tokenizer

Kimi K3's published tokenizer stands in for K3.1 Pro's, whose weights are not
out: it counts 6.8% under the billed replies, ~42 tokens a reply, likely the
reply markup. Against Flash-Next's, on the traces' text (K3's: tiktoken,
163,584 ranks, digits grouped by up to three; Flash-Next's: 248,077, digits
one by one):

- Flash-Next needs 1.03x Kimi's tokens on the system prompt, 1.21x on user
  prompts and tool results (grids, arrays, coordinates), 1.10x on reasoning
  and 1.05x on replies and tool calls.
- A Kimi token boundary is a Flash-Next one too at 98-99.7%, so a Kimi token
  is nearly always a whole run of Flash-Next tokens; tokens map 1:1 in
  89-93% of reasoning, 88-90% of code and ~80% of tool results.
- Boards printed as letters map 1:1 at 99%, against 56% printed as hex: every
  trace set from the re-arc run on prints letters, the seal's default since.
- What stays split: two-digit numbers, punctuation before a newline,
  contractions (" I'll"), " magenta", MOUSE.

## The API

Measured on `kimi-k3.1-pro`, 2026-10-27 to 10-29. The API's own differences
from OpenAI's are in `ullm.backends.kimi`'s docstring.

- **Limits.** The account takes 50 requests at once, 200 a minute and 3M
  tokens a minute; past 50 a 429 says `rate_limit_reached_error`. Heavier
  use for a while draws a throttle that does not say so: after an hour and a
  half at ~100M prompt tokens an hour (20 games), every request is held about
  a minute and refused as `engine_overloaded_error` for hours, while the
  same model serves another account (a paired request through OpenRouter,
  2026-10-29). An hour without requests brought service back. The backend
  retries a call for twelve hours and the launcher's runs wait that out
  (`--game-hours` 24 allows for it); each retry's line in `stdout.log` says
  what the API answered.
- **Cache.** The previous prompt is hit in 2,048-token blocks, never the
  previous reply; images keep it. Calls ask for the hour-long lifetime,
  whose writes cost twice, so a call held by the throttle keeps its game's
  cache. List prices: $2 a million uncached prompt tokens, $0.20 cached,
  $10 output.
- **OpenRouter** routes the model to Moonshot itself, but returns logprobs
  for the answer's tokens only, none for the reasoning: not used.
- **The job.** A run is mostly waiting: 159 games averaged 1.4 cores, so 16
  CPUs; 38 games at once held 11 GiB, so 64 GiB.

Using the API traces for SFT within the Kaggle competition is settled
(2026-10-29): it does not compete with Moonshot's services, which
their API terms (3.2(5)) restrict.

## Trace sets

ARC3, under `/data/shared/research/arc_3_results/shared/distill/kimi-k3.1-pro/`.
All but the first two play with the world-model default; every set from
2026-10-28 on records logprobs. "Cut" games were stopped in play.

| Run dir | Games (traces) | Won | Output tokens (reasoning) | Boards | Logprobs | Bundle |
|---|---|---|---|---|---|---|
| `seal-smoke-20261027-170835` | tu93 (1) | 1 | 0.08M (0.06M) | hex | no | 0.2 MB |
| `seal-smoke-20261027-180714` | 38, stopped at the start (4) | 0 | 0.01M | hex | no | <1 MB |
| `seal-smoke-20261027-181442` | 38, stopped at ~30 min (7) | 3 | 0.17M (0.12M) | hex | no | 1 MB |
| `seal-smoke-20261027-184543` | 38 (38) | 36 | 4.60M (3.53M) | hex | no | 16 MB |
| `seal-smoke-20261028-173617` | pathfinder (1) | 1 | <0.01M | hex | yes | 1 MB |
| `seal-re-arc-rest-s20261002-20261028-214737` | 159, 31 cut at 11 h (159) | 128 | 4.33M (3.09M) | letters | yes | 583 MB |
| `seal-smoke-s20261003-20261029-062616` | 38, cut at ~30 min (38) | 1 | 0.25M | letters | yes | 36 MB |
| `seal-smoke-s20261004-20261029-062641` | 38, cut at ~30 min (38) | 1 | 0.22M | letters | yes | 33 MB |
| `seal-smoke-s20261005-20261029-062712` | 38, cut at ~20 min (38) | 0 | 0.23M | letters | yes | 34 MB |
| `seal-smoke-s20261003-20261029-065606` | 38, cut at ~2 h (38) | 7 | 1.72M (1.26M) | letters | yes | 229 MB |
| `seal-smoke-s20261004-20261029-085243` | 38, throttled throughout (20) | 0 | 0.05M | letters | yes | 8 MB |
| `seal-smoke-s20261005-20261029-110500` | 38, cut at 3.6 h (34) | 14 | 2.12M (1.59M) | letters | yes | 282 MB |

`seal-re-arc-rest-…/partial_traces/` holds hourly copies of the finished
games' traces taken while it ran; the bundle supersedes them.
