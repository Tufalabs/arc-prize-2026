# Distillation teachers

*Register: living.*

Kimi K3.1 Pro played through Moonshot's API, to record TAAF traces a
Qwen3.8 Flash-Next student is SFT-trained on (one student for ARC2 and
ARC3; the ARC3 teacher runs live in arc3-platypus's `tools/teachers/`,
whose README has the API and the traces in more depth).

| File | What |
|---|---|
| `teachers.py` | The teacher entry: model, key variable, the factors behind its accounting, its slots |
| `duck_smoke.py` | The launcher: the duck on the teacher's API over TAAF's named task sets, as a CPU job; `--traces` records |

The key is `KIMI_API_KEY` in the launching shell's environment. A launch is
`uv run python tools/teachers/duck_smoke.py --api kimi-k3.1-pro --set
arc2-train-hard,arc2-eval --traces`; the module docstring lists every flag.
Runs go under `/data/shared/research/arc_2_results/<user>/distill/kimi-k3.1-pro/`.

## The student

A model trained on these traces plays the duck they were recorded on:

    DuckAgent.build(engine="qwen3.8-flash-next", environment=..., kimi_student=True)

`kimi_student` sets `duck.agent.KIMI_STUDENT`, the model-visible settings
the teacher played with that the column has off: summaries written after
the model's own reasoning. The teacher's launcher builds from the same dict.
What else the teacher sets is for its API and does not carry: the
estimate-mode accounting and its budgets in the teacher's tokens, no banned
token in a summary, no host watchdog. The reasoning level stays the column's
`xhigh`, the level the traces are rendered at in training. Sampling stays
the column's too; whether a student should sample as the teacher was
sampled (temperature 1, top_p 0.95) is open, in `roadmap.md`.

## The traces

`--traces` records at TAAF's hosted-API grade (`taaf.trace.v2`), one trace
per task, bundled into the job dir at teardown:
`retained_traces/trace_bundle.tar.zst`, with `trace_manifest.json` mapping
each run id to its member prefix. Each step is one model call: the whole
message list it was sent, past reasoning included, and `output_message`
with the reasoning and the tool calls. From this pin on a step also carries
the API's top 20 per-token alternatives in `extra["teacher_logprobs"]`, in
Kimi's tokens; the sets below predate it.

## The API

The account takes 50 requests at once, 200 a minute and 3M tokens a minute.
Heavier use for a while draws hours of refusals that read as the API's own
overload (`engine_overloaded_error`) while the same model serves other
accounts. The backend retries a call for twelve hours, and the launcher's
runs wait that out: no host watchdog, and each task's wall cap raised by the
retry window. Each retry's line in `stdout.log` says what the API answered.

## Trace sets

Under `/data/shared/research/arc_2_results/shared/distill/kimi-k3.1-pro/`.
"Submitted" tasks gave a wrong answer.

| Run dir | Tasks | Won | Submitted | Crashed | Output tokens (reasoning) | Bundle |
|---|---|---|---|---|---|---|
| `arc2-train-easy-20261028-080042` | 10 | 10 | 0 | 0 | 0.04M (0.03M) | 0.1 MB |
| `arc2-train-hard+arc2-eval-20261028-085042` | 167 | 143 | 18 | 6 | 3.89M (3.16M) | 7 MB |
| `arc2-train-hard+arc2-eval-20261028-120605` | the 6 crashed, again | 4 | 2 | 0 | 0.23M (0.20M) | 0.4 MB |
