# ullm

*Register: living.*

The platypus LLM layer: a backend-agnostic conversation framework with
peer backends (in-process vLLM, OpenAI-compatible, native Anthropic,
stubs), turn/response managers carrying the malformed-output and
context-length defenses, Python-kernel and file tools, and the
capture / replay / R5-trace seams. A hard fork of the standalone ullm
repo, keeping the import name `ullm`.

Docs here:

- `design.md` — the architecture map
- `kv_offload.md` — the CPU KV tier: design, defaults, how it was tested
- `flash_optimization.md` — serving Flash-Next: the experiments behind both deployment cells, and the open Flash work
- `flash_optimization_arc2.md` — its ARC2 addendum: the four-L4 cell, the images ARC2 sends, and ARC2's B200 readings
- `traffic_benchmark.md` — the traffic benchmark: throughput read from recorded traffic replayed against a live engine; how to run it and what a reading means
- `model_checklist.md` — what to do and check before a model does its best on ARC2 and ARC3, on the cluster and on Kaggle
- `AGENTS.md` — ullm-specific coding standards and test tiers
- `roadmap.md` — open fork work
