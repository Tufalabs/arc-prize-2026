# duck

*Register: living.*

The duck agent — a CodeAct solver on the platypus contract: frame
segmentation through `frame.segment()`, native `python` and `memory`
tools, a preloaded Python `submit_action(...)` function, an aligned
frames/actions/animations timeline, a full context compacted by a summary the
model writes (`design.md`, "Summarize") or by eviction, with exact token
counts, and reasoning round-tripping.

Originally a behavior-faithful port of ARC3-Inference's `ToolAgent` and
solver loop (`c7bf328`); that system was the reference the duck was judged
against until the decoupling, and comparing the two is no longer something
we do. What the port established survives as design and as measurements,
not as a ledger of deltas.

Some of those remain fields rather than inseparable implementation
choices. Focused comparisons can address colors as indices with
`frame_api="numeric"`, hide UNDO with `expose_undo=False`, and disable
only the completion-cutoff wording with `completion_cutoff_notice=False`;
Python lifetime remains the independent `persistence` setting. The prompt,
tool description, accepted actions and actual tool list follow every
combination. Host-call validation, aligned internal bookkeeping and
retained-history byte identity remain shared invariants rather than
ablation controls. The switches describe supported model-facing
interfaces, not isolation from Python reflection; parent callbacks remain
hardened regardless of mode.

Where the behavior came from. It is chosen against measured play on
the official game set, not inherited from the original, so several pieces
diverge from it: the journal rather than prose extraction, a REPL that lives
for a level, a tool-output cap generous enough that a grid dump survives
intact, a reply cap set to let a long turn finish rather than to keep the
window small, mid-turn round pruning rather than the whole-turn drop the engine
otherwise forces, and no attached grid image — the board reaches the model
where it can be worked on, inside the kernel, and the coarser second copy in
the prompt did not earn its tokens.

Deployments: `DuckAgent.build(engine=…, environment=…)`, one call per
coordinate — `engine` selects the checkpoint (`"qwen3.8-flash-next"` by
default, the pruned Flash-Next derivative; `"qwen3.8-27b"` for
Qwen3.8-27B-FP8, and `"qwen3.6-27b"` for the previous model), `environment` the box
(`"slurm-b200"` by default, `"kaggle-rtx"` for the offline Kaggle stack).
An existing engine can be supplied as `backend_factory` in either environment;
set `derive_max_model_len=False` and `derive_max_num_seqs=False` explicitly.
`serving="server"` picks the OpenAI-server path over the same engine args, for the 27Bs only: Flash-Next is refused there (`packages/ullm/roadmap.md`, "Park mode over the server topology"). Everything else is a field.
Run via `platypus.LLMAgentSolver(DuckAgent.build())` under a taaf benchmark —
see the root `agent_authoring.md` for the contract and `settings.md` for
where each setting lives.

## Reading it

`agent.py` is the agent, written to be read top to bottom: its module
docstring carries the reading order and the map of what the duck asks the
harness for. Everything else is what would have made that one file harder
to read.

- `prompts.py` — every fixed string the model reads, on `DuckPrompts`:
  wording as constructor-settable fields, composition as methods; what the
  agent decides arrives per render as a `PromptModes`
- `state.py` — the typed per-game bookkeeping (timeline, batch, summary,
  journal entries); the model-visible wire dicts are built from it at
  the serialization seam
- `kernel_code.py` — the sandbox prelude the model's Python runs under;
  `segmentation.py` is spliced into it verbatim
- `python_tool.py` — the `python` tool: syntax check, kernel dispatch,
  and the original's tool-result rendering
- `memory_tool.py` — the native `memory` tool; it appends one required `text`
  finding to the host-timestamped journal, which retains about 1,000 recent
  tokens by evicting oldest whole entries. A frozen copy is placed at the
  system-prompt tail and refreshed only at conversation compaction; each newer
  entry is already visible in its immutable tool call, and the result returns
  only a compact host-timestamp acknowledgement
- By default, `submit_action(...)` is a preloaded function inside the stateful Python
  session, not a native tool or module; call it directly without importing it
- The default Python surface (`wide_sandbox`) can import `bisect`, `collections`, `copy`, `difflib`,
  `fractions`, `functools`, `hashlib`, `heapq`, `importlib`, `inspect`, `itertools`, `json`, `math`,
  `numpy`, `operator`, `pickle`, `random`, `re`, `scipy`, `statistics`, `string`, `textwrap`, `time`
  and `traceback`, each also bound by its name in a new session, and NumPy as `np`; `open` and every
  exception class are builtins there. `wide_sandbox=False` takes away `hashlib`, `importlib`,
  `inspect`, `pickle`, `scipy`, `time`, `traceback`, `open` and the exception classes the narrow
  list does not name
- `frame.ascii` is a newline-delimited string. `frame.segment()` nodes expose
  `pixels` as `[row, col]` coordinates and `pixel_count` as the integer area;
  `frame.array` is a cached 2-D NumPy array of the same letter-coded colors
- `presets.py` — what ARC3 asks of a deployment, as layers over ullm's
  engine/environment coordinates
- `traffic.py` — `TrafficDuck`: the column's engine measured on recorded
  traffic instead of games (`packages/ullm/traffic_benchmark.md`); the
  launcher, the workload cutter and the archive are under the root `tools/`
- `action_names.py` / `grid_utils.py` — the action labels the model is
  taught, and the ARC color legend

## Docs here

- `design.md` — how the duck is put together, and the reasons that span
  more than one place
- `evidence.md` — the measurements a current default rests on, and what
  each does and does not settle
- `roadmap.md` — what we intend to change, and decisions still open
