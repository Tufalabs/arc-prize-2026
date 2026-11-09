# Flash RL training

How RL training of Qwen3.8-Flash-Next works here and why. Environment and model-stack details are in
[docs/environments.md](../docs/environments.md) and [training/flash/README.md](flash/README.md);
numbers in the [reference measurements](#reference-measurements).

## What this is

Qwen3.8-Flash-Next is a 48-layer hybrid MoE (`qwen4_exp`): 36 Gated DeltaNet (GDN) linear-attention
layers, 12 sparse-attention (QSA) layers, top-10 routed experts plus a shared expert per layer, a
per-layer-embedding (PLE) n-gram table and a vision tower. We train **e307**, the NVFP4 checkpoint
pruned to 307 experts per layer that the harness also deploys
(`/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform`, see
[docs/flash_pruning.md](../docs/flash_pruning.md)), with LoRA on dense projections
(`training.weight_mode = "flash"`), through the same RL trainer, collector, harness and parity gate
as the 27B recipes. Unlike the 27B LoRA path, the learner is this repository's Flash stack
(`training/flash/`, `FlashPPOTrainer`) on four torchrun FSDP ranks rather than a PEFT model; serving
uses the harness's vLLM fork in its own environment rather than vLLM 0.25.1, started by the job
script; only dense projections carry adapters, and the servers never see them: every publication
ships the policy as merged dense weights, applied in place on engines without LoRA; and the parity
gate has its own thresholds, because the Flash engine's floor is higher. The models share a tokenizer, so token-id settings such as
`agent_summary_forbidden_token_ids = [248058]` (`<tool_call>`) carry over.

## Process layout and allocation

### Process split

- **Learner**: `python -m training.launch_frozen` freezes source and config into the run directory;
  torchrun then starts the ranks (`training.flash_entrypoint`) in `ARC3_FLASH_PYTHON`. It never
  imports vLLM (the `without_vllm` import probe enforces this): it reaches the servers over HTTP,
  broadcasts merged policy weights from its rank 0 into the servers' GPUs over NCCL
  ([Merged policy weights](#merged-policy-weights)), and spawns the rollout worker
  (`scripts/platypus_rollout_worker.py`) in `inference.platypus_venv`.
- **Rollout servers**: one vLLM engine per GPU (tensor parallel 1), started by the job with
  `scripts/launch_platypus_server.sh` and fronted by `training.native_vllm_server`. Official eval
  is refused in flash mode: its servers would need an NCCL group of their own with the learner.

### Allocations

`scripts/flash_rl_submit.sh` composes the Slurm request for `scripts/flash_rl.sbatch`. **Shared**
(`--server-nodes 0`) is one 8-GPU node: servers on GPUs 0-3 (`inference.server_gpus`), learner on
4-7 (`training.flash.device_ids`). **Heterogeneous** (the default) makes component 0 the learner
(`--learner-gpus`, normally 4) and each further component a group of `--server-nodes` nodes with
`--server-gpus-per-node` servers, repeated `--server-groups` times; no learner collective crosses
nodes. The job sets `inference.server_allocation` to `separate` with `inference.base_urls` from
Slurm hostnames and unique ports, and renumbers both GPU fields from the allocations: the learner
takes GPUs 0..N-1 of its node (N = `--learner-gpus`) and each server node 0..M-1 (M =
`--server-gpus-per-node`). Those are the allocation's own GPUs because kander confines each job step
to its GRES devices (cgroup `ConstrainDevices=yes`) and numbers them from 0; the learner and each
rollout node refuse to start when `nvidia-smi -L` shows them another count. In this layout the
config's `inference.server_gpus` and `training.flash.device_ids` are placeholders that need no
editing, and the claims are checked after the renumbering (`launch_config.json`); only the shared
layout reads them as given, as disjoint indices on one node. Servers bind `0.0.0.0` behind a fresh
per-segment API key passed only through the environment. Publications take the same path in both
layouts: the servers' workers join the learner's NCCL group at the learner node's name and a free
port, and every publication goes over it (InfiniBand between nodes, NVLink or shared memory within
one), so nothing goes by path between allocations. Each rollout node copies the checkpoint to
scratch once for its engines (four engines loading it concurrently over NFS took 13 minutes each).

Before writing `launch_config.json`, the heterogeneous launcher checks up to 40 port blocks,
starting at `50000 + SLURM_JOB_ID % 4000` and stepping the base by 101 after a failed probe.
It binds each gateway port (`base + 100*i + 5000`) on the learner, then uses one short
`srun --het-group=<group> --nodes=1 --ntasks=1 --overlap --nodelist=<node>` per server node to
bind that node's engine ports (`base + 100*(node_index*gpus_per_node + gpu)`). Both probes use
`SO_REUSEADDR`, close the test sockets, and never reclaim a listener. Only a block free on all
nodes becomes the launch config's `inference.base_urls` / `SERVER_URL_LIST`; gateway ports move
with the same base. These checks do not reserve ports: each rollout node still checks immediately
before starting an engine and exits 1 with `engine port <port> is in use on <node>` on failure.
On 2026-10-12 at 01:50 CEST, job 2937399 stopped before its learner started because engine port
51399 was occupied on dgx01; the learner-only gateway probe had missed that collision.

### Sizing and scheduling

The recipe collects 128 attempts per window (16 puzzles x 8 attempts, the GRPO group) on four
servers, 32 in flight on each. Per-server concurrency is set by `parallel_rollouts` divided by the
number of servers (a recipe field, not a launcher flag). The job derives each engine's
`--max-num-seqs` from the run's frozen config at every segment (`training.job_shape`):
`ceil(parallel_rollouts / servers) + 8` (40 here), a server's share of the window, at most
`inference.max_concurrent_episodes` (a per-server cap; unset admits every slot), plus headroom for
the trainer's control-plane warm-up draws. An exported `SERVER_MAX_NUM_SEQS` overrides it; the job
refuses one below the share, since an engine with fewer slots queues the window's episodes and
collection slows without an error. The server launcher has no default.
`parallel_rollouts` must stay a multiple of `sampling.game_sampling_games_per_window`.

At the recipe's deployment context the KV pool, not the slot count, is what bounds a server. A
drafting engine's pool is 3,967,385 tokens of KV in attention blocks of 928 tokens, 4,275 blocks
([Speculative decoding](#speculative-decoding-opt-in); measured at a 49,152-token engine window,
and vLLM sizes it from free memory, so the 262,144 window should leave it unchanged), and a context
holds whole blocks. A segment at its longest, the window plus a reply (98,304 tokens), takes 106
blocks, so the pool holds 40 of them: the 32 attempts in flight fill at most 3,392 blocks (79%), and
all 40 slots the job derives at most 4,240 (99%; the eight beyond the share serve short
control-plane draws). At the window alone (73,728 tokens, 80 blocks) it holds 53 sequences, at the
Duck's own summary bound (125,952, 136 blocks) 31. Contexts run as a sawtooth between a summary's
reset and the 48,640-token prompt budget, so occupancy stays below that worst case; it is not
measured at this window. A full pool makes the engine preempt and recompute a request, which is
slower, not wrong. Concurrency stays at 32 per server.

The window is 128 attempts rather than 64 (16 per server) because servers were far from saturated
([Serving](#serving)) and per-server concurrency is the cheaper lever. At 64 the run was
rollout-bound: a pass over a window took the learner about 6.5 of ~16 minutes, extra train cycles
filled the rest, and the learner could have kept pace with about 2.8 server GPUs per learner GPU. At
32 in flight per server decode throughput is about 1.84x (benchmark: 890 tokens/s at 16 concurrent
requests, 1,638 at 32), which the learner can absorb within the window; 64 per server (about 3x)
would leave the `auto` sample budget training a smaller share of each window. Collection time and
trained share at 128 are not measured yet.

With `training.rollout_schedule = "overlapped"` (the default) the servers collect window N+1 under
the adapter live at its start while the learner trains on window N, so neither idles through the
other's phase; the price is data one update old (a larger lag stops the run), which the PPO ratio
and a lag-aware [parity check](#parity-gate) handle. Collection stops after the final window.

## Environments and launching

| Environment | Runs | Built from |
|---|---|---|
| `trainer-flash-distillation-<key>` | the learner | this repository's lock (torch 2.13.0, Transformers 5.16.1, FLA 0.5.1, Triton 3.7.1), plus the harness's `ullm` without dependencies |
| `harness-<key>` | every server; ARC-3 rollout workers | the harness lock (`platypus` submodule, arc3-platypus `e53d822d`): the vLLM fork with its torch and FlashInfer |
| `arc2-worker-vllm-fork-<key>` | ARC-2 rollout workers | `environments/arc2-worker-vllm-fork`: duck, platypus, taaf from `arc2-platypus` (`f380ffd`), ullm from `platypus`, the harness lock's serving stack |

The stacks cannot share an interpreter. The learner's torch, FLA and Triton are qualified apart from
the fork's, so a serving upgrade never silently changes the learner (this repository's own `harness`
extra pins the 27B engine and is not for Flash), and the ARC-2 harness reuses the package names
`duck`, `platypus` and `taaf`. The ARC-2 worker must still match the engines' image-processing
identity (ullm's `EngineProxy` refuses to render when Transformers, tokenizers, torch, torchvision,
Pillow or NumPy differ): its lock is checked against the harness lock and the job compares the two.

Both harness pins are their `main` heads of 2026-10-09 (arc2-platypus `f380ffd`, arc3-platypus
`e53d822d`, both serving fork `3bda1ba09`); runs before that used arc2-platypus `aee9d37`,
arc3-platypus `a5196391` and fork `077920f3`. The ARC-2 Duck at `f380ffd` plays with the settings
its sweeps 8 and 9 chose as class defaults (arc2-platypus `c30bb92`): an audit turn before a game
stops, file tools (`view_file`, `str_replace`, `create_file`), changed answers shown as an image,
files and REPL listed after a summary, a more thorough summary request; the worker keeps retained
failed draws and no malformed-reply nudge (that nudge drops a reply the backend refused; training
refuses none, so a retained empty reply gets the analyzer entry's ordinary nudge instead, unless
`agent_terminate_on_empty_reply` ends the game on it). Rollouts therefore differ from those of runs at the
earlier pins, and such a run cannot resume on these environments (the job refuses other environments
than the ones a run started with). Rollouts serve the prompt injection the Duck's engine preset
resolves, which training no longer overrides: Flash serves no opening text (its `qwen3.8-flash-next`
engine layer serves none: an opening halves what this model reasons per action, and its runs were
played without one), and the 27B serves the class defaults, `We need solve` on a game's first prompt
and `We need` on every later one (schedule `game`), where training used to serve `We need solve` at
each context and nothing later, a change to 27B training behaviour. Every run reports the three
settings its agent served; the checkpoint identity pins the first window's (`harness_agent`,
`training/rollout_identity.py`) and a later window or a resume reporting others stops the run naming
the setting ([docs/arc2_training.md](../docs/arc2_training.md)).

### Building and submitting

On the kander login node, in the layout of our training repository (see
[Building](../docs/environments.md#building) for this monorepo):

```bash
git clone --recurse-submodules <training-repo-url> /data/personal/$USER/arc-training
cd /data/personal/$USER/arc-training
scripts/build_envs.sh        # or use a prebuilt root: export ARC_TRAINING_ENVS=<artifact root>
scripts/flash_rl_submit.sh --config configs/flash_8gpu_arc2_grpo_tokmean.json
```

`scripts/build_envs.sh` submits the three environment builds (`trainer`, `harness`, `arc2-worker`;
see [docs/environments.md](../docs/environments.md)) for HEAD (`--revision`) into
`/data/personal/$USER/arc-training-envs` (`--out` or `$ARC_TRAINING_ENVS`), skips a recipe already
built for the same locks and pins, and prints the job ids and the submit command to run once they
have completed (with `-- <repo> <commit>` when that is not the checkout's HEAD). `--check` only
reports which of the three are built for the revision (exit 3 if any is not). The revision need not
be pushed (the jobs archive it from the checkout), but its submodule pins must be on their remotes.

`scripts/flash_rl_submit.sh` then runs HEAD of its checkout: the job archives that commit, so the
helper refuses uncommitted changes (`--allow-dirty` runs HEAD without them; `-- <repo> <revision>`
runs another clone's revision). Each `ARC3_*_VENV_TARBALL` left unset becomes the newest artifact
under the root (`--envs`, `$ARC_TRAINING_ENVS`, else the personal one) built for the revision's
locks and pins (`scripts/resolve_envs.py`); explicit variables win, and a missing environment ends
with the nearest artifact's differing fields and the exact `build_envs.sh` command to run (carrying
the submission's `--exclude`). The job gets the revision's full commit. The helper creates the run
directory `/data/personal/$USER/arc-training-runs/<config stem>-<YYYYMMDD-HHMM>/lora` (`--run-dir`),
printing each choice; server concurrency is the job's ([Sizing](#sizing-and-scheduling)). The
default shape is the heterogeneous one above (`--learner-gpus 4 --server-nodes 1
--server-gpus-per-node 4`); `--server-nodes 0` uses one shared node.

The helper asks for B200s with 12 CPUs and 96G per GPU, six hours (`--time`) and a TERM ten minutes
before the limit. `--test-only` has Slurm validate the request (creating no run directory), and
`--dry-run` allocates, stages and prints every command without starting engines or training (it
writes a runtime fingerprint: use a throwaway run directory). ARC-2 configs also stage the worker
environment: it becomes `inference.platypus_venv` on the learner, the job copies the two pinned
ARC-2 datasets from its taaf into the archived source, and servers always take images. ARC-3 configs
need no worker environment: their workers run from the harness; images follow
`inference.agent_multimodal_context` and need the `group_mean` baseline
(`harness_support/arc3/trainer.py` refuses them with the hidden critic). Servers take
`inference.agent_max_served_sequence_tokens` (262,144 for the recipe, the checkpoint's native window)
as their maximum length.

### The recipe and smoke configs

The one shipped recipe, `configs/flash_8gpu_arc2_grpo_tokmean.json`, began as the 27B ARC-2
token-mean recipe (`configs/grpo_6gpu_arc2_overfit.json`) with `agent_terminate_on_empty_reply` on
(off since 2026-10-10, below) and only Flash changes: model and allocation, four servers, official eval off, LoRA dropout 0, the
GDN `in_proj_b`/`in_proj_a` targets, and 16 puzzles per window instead of 8. That part stands: GRPO,
token-mean loss, LoRA r16/alpha 32 on twelve dense targets, lr 5e-6, routers trained (`train_routers`, FP32 masters at the base learning rate), dithered
publications, the Flash parity thresholds, sampling at temperature 1.0 and top-p 1.0, 128 attempts
(16 puzzles x 8) per window. On 2026-10-09 it replaced the first Flash recipe (the run
`arc2-flash-grpo-tokmean-term-xhigh-compact-20261008` keeps that one in its frozen config) in six
places, so that the run trains what the deployment plays, on the puzzles the base model does not
already solve, each seen differently from window to window:

`configs/flash_8gpu_arc2_grpo_epmean.json` is the same recipe with `training.loss_aggregation: episode_mean` (the weighting of the first 27B ARC-2 run, `arc2-overfit-terminal-epmean-20260916`); `tests/test_flash_arc2_recipe.py` keeps every other value identical to the token-mean recipe.

| | First recipe | Now | Why |
|---|---|---|---|
| Puzzles | all 120 of `arc2-eval` | 89 listed in `inference.games`, from `arc2-eval` (52) and `arc2-train-hard` (37) | the puzzles e307 did not complete on all ten TAAF passes (below) |
| Context | 32,768 window, 4,096 reply cap, 0 margin, 49,152 served | the deployment Duck's: 73,728, 24,576, 512, 262,144 (below) | rollouts at the lengths the harness plays |
| Game limit | 15 minutes | 40 minutes, still no token cap (`agent_max_episode_tokens` null) | more of each longer game fits; the window is the game limit. The first run at 20 minutes (2026-10-10) gave each episode about 50k completion tokens at 32 games per server, below the median the base model needs to solve these puzzles in TAAF (46k, p75 79k), so 40 minutes doubles the budget |
| Drafting | off | on, ullm's e307 drafter and 4 drafts (the ARC-2 column deploys REAP-307 with its own) ([Speculative decoding](#speculative-decoding-opt-in)) | about 1.5x decode at 32 requests; the sampled distribution is unchanged |
| Augmentation | off | `inference.augmentation` on, dihedral and colors ([Training signal](#training-signal-details-that-matter)) | each window draws one dihedral transform and color permutation per puzzle, the same for its 8 attempts, so 89 puzzles do not repeat verbatim |
| Learner | 40,960-token micro-batches, 49,152 ceiling | 98,304 (window + reply cap), 262,144 ceiling, `offload_every` 4 | segments reach the window plus a reply ([memory](#learner-memory-at-the-deployment-context)) |

**Empty replies** (2026-10-10). A reply with no text and no tool call no longer ends the episode:
the recipe sets `agent_terminate_on_empty_reply` false, so the Duck nudges past it as the deployment
harness does (ullm's "Malformed reply: no text and no tool call." in the turn, then the analyzer
entry's ordinary nudge) and the game goes on. An empty reply is a malformed call: each one costs
`training.malformed_call_penalty` (0.05), as it always did. With termination on, 39% of the run
`arc2-flash-grpo-lt100-aug-20261010`'s episodes ended on an empty reply. Nothing but the 40-minute
game limit and the Duck's 5,120 requests a game caps how many a looping episode makes (the recipe
has no token cap), and each is a trained step. The reward caps what they cost:
`training.max_episode_penalty` 0.5 limits an episode's summed penalty, so the reward is
correctness - min(0.5, 0.05 x malformed calls) and a solved puzzle with many penalties still scores
above an unsolved one (2026-10-10). The checkpoint identity pins the cap.
`rollout/avg_empty_replies_per_rollout` counts the empty replies among `rollout/avg_malformed_calls`
and `rollout/penalty_cap_fraction` the episodes charged the cap
([docs/arc2_training.md](../docs/arc2_training.md)).

**Crashed games** (2026-10-11). Training now uses the harness's own
`on_agent_error="crash_game"`, platypus's default and every deployment
column's: a game whose agent raises ends `crashed`, is scored from its standing
answer (correctness of what it submitted, penalties and the cap as usual) and
trains as a `CRASHED` terminal, and the window goes on. It used `"raise"` until
2026-10-11, which ended the 40-minute run `arc2-flash-grpo-lt100-aug-40min-20261011`
(job 2932277) on one Duck kernel bug in one game of 239 (a kernel housekeeping
failure in the Python tool); at that rate about 40% of 128-episode windows would
have died. `rollout/crashed_count` and `rollout/crashed_fraction` count them, and
a window where every run crashed is still refused
([docs/arc2_training.md](../docs/arc2_training.md)).

**The puzzles.** `inference.dataset` names two TAAF sets, `["arc2-eval", "arc2-train-hard"]` (the
47 `arc2-train` puzzles e307 did not solve on both of two passes, taaf's `HARD_TRAIN_TASK_IDS`), and
`inference.games` lists the 89 played, in the order of
`kander:/data/personal/$USER/flash-bench/tools/trainset_base_lt100/kept_game_ids.json`
(`kept_table.md` beside it has each puzzle's numbers). e307 played both sets ten times with TAAF on
2026-10-09; the 78 puzzles completed on all ten passes (every test output right in the standing
answer) were dropped. Of the 89, 10 were never completed, 13 in one to three passes, 19 in four to
six and 47 in seven to nine; nine multi-output puzzles are kept although every pass got at least one
output right. A section may name one set or several and list games from their union (empty: all of
it); the sets must be disjoint and a listed game outside the union is refused when the trainer
starts (`harness_support/arc2/trainer.py`, `_selection_manifest`). The checkpoint identity records
every set's manifest and the game list; one set played whole keeps that set's manifest, as before.
The trainer reads only the packed sets, so the derived set's ids are copied into
`harness_support/arc2/dataset.py` (a test compares them with taaf). This is still an overfit run:
it trains on the puzzles it is scored on and holds none out, and 52 of them are `arc2-eval`
puzzles, so `arc2-eval` scores of its adapters are not held-out scores.

**The context** is what `DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")`
plays with at arc2-platypus `f380ffd`, derived as the Duck derives it (lines in
`arc2-platypus/packages/duck/src/duck/`; `tests/test_flash_arc2_recipe.py` reads them back):

| Field | Value | From the Duck |
|---|---:|---|
| `agent_context_window` | 73,728 | `DuckAgent.context_window_tokens` (`agent.py:240`): the 48,640-token prompt budget the sweeps read at, plus the reply cap and 512 |
| `agent_max_completion_tokens` | 24,576 | `reply_cap_tokens` (`agent.py:575`) |
| `agent_safety_margin_tokens` | 512 | `safety_margin_tokens` (`agent.py:522`) |
| `agent_compact_to_tokens` | 48,640 | W - reply cap - margin: every ARC-2 column summarises (`presets.py:187`), and under `summarize` the reset ceiling is the prompt budget (`_observer_compaction_target`, `agent.py:3602`; `_context_budget_tokens`, `agent.py:3497`) |
| `agent_max_summary_tokens` | 9,216 | 2 x (W // 16): the summary target is W // `compaction_ratio` (`agent.py:2456`, 16 at `agent.py:270`), its cap twice that (`_summarize_live_history`, `agent.py:2780`) |
| `agent_max_served_sequence_tokens` | 262,144 | the engine window the Duck summarises under: `engine_ceiling_tokens = engine_args.max_model_len` (`presets.py:201`), Flash-Next's native window in ullm's deployment (`platypus/packages/ullm/src/ullm/backends/deployment.py:327`, `_flashnext_config`). It holds the Duck's own bound, W + reply cap + 6 x (W // 16) = 125,952 (`_derived_engine_window`, `agent.py:1011`), and ours, W + summary cap + 12,288 = 95,232 |

Not taken from the Duck: its play sampling (temperature 0.8, top-k 194, `presets.py:132`; training
samples the distribution the learner scores), its 500,000-token cap per game on this column
(`presets.py:359`; a 40-minute game generates far fewer), its eight-hour wall net (`presets.py:90`)
and its checkpoint (the column serves the REAP-307 prune; RL trains e307).

**The learner** packs each context into segments as long as their last request: a prompt within the
48,640 budget and a reply of at most 24,576, or a summary request, which carries the whole live
history past the budget (the last reply and the tool results after it) and a summary of at most
9,216. Segments therefore reach about the window plus a reply, 98,304 tokens, and `max_micro_batch_tokens` holds one such segment; a longer one trains alone,
so the peak is the longest segment's either way. `max_sequence_tokens` must be at least the served
ceiling (loading refuses less): it bounds, it does not allocate. A 98,304-token micro-batch does not
fit a B200 without activation offload; `training.flash.offload_every` 4 fits it with 14.5 GiB to
spare at a cost of 3-4% ([Learner memory](#learner-memory-at-the-deployment-context)).

Launchers take whole config files, so a smoke run uses a derived copy, passed by absolute path (the
job reads it there, not from the revision); `parallel_rollouts` must stay a multiple of the puzzles
per window, with at least two attempts each:

```bash
python3 -c 'import json, sys
config = json.load(open(sys.argv[1]))
config.update(num_iterations=2, parallel_rollouts=16)
config["sampling"]["game_sampling_games_per_window"] = 2
json.dump(config, open(sys.argv[2], "w"), indent=2)' \
  configs/flash_8gpu_arc2_grpo_tokmean.json /data/personal/$USER/flash-arc2-smoke.json
scripts/flash_rl_submit.sh --config /data/personal/$USER/flash-arc2-smoke.json
```

Fewer GPUs are submit options, not config edits (`--learner-gpus 2 --server-gpus-per-node 2`; see
[Allocations](#allocations)).

**Shrinking the lengths.** A smoke run may shorten the context; the lengths then move together, and
the reply cap with them (it must stay below the window less the margin and 1,024). The 32k column
is the first recipe's lengths, whose learner memory is measured ([Expert storage](#expert-storage)).
Loading checks the summary cap, the served ceiling (at least the rule below; unset fails) and the
learner ceiling; the compaction fields are checked when collection starts.

| Field | Rule | Recipe | 32k window |
|---|---|---:|---:|
| `inference.agent_context_window` | the window W | 73,728 | 32,768 |
| `inference.agent_max_completion_tokens` | the reply cap R | 24,576 | 4,096 |
| `inference.agent_safety_margin_tokens` | the margin M | 512 | 0 |
| `inference.agent_compact_to_tokens` | the compaction trigger, W - R - M | 48,640 | 28,672 |
| `inference.agent_max_summary_tokens` | 2 x (W // 16): the ARC-2 Duck's summary cap, which its worker requires | 9,216 | 4,096 |
| `inference.agent_max_served_sequence_tokens` | at least the larger of W + summary cap + 12,288 and the ARC-2 Duck's own bound, W + R + 6 x (W // 16) | 262,144 | 49,152 |
| `training.max_sequence_tokens` | at least the served ceiling | 262,144 | 49,152 |
| `training.max_micro_batch_tokens` | tokens packed into one micro-batch, bounding learner memory; a longer segment trains alone | 98,304 | 40,960 |
| `training.flash.offload_every` | offload every Nth layer's checkpoint inputs to the host; 0 where the longest segment fits without ([memory](#learner-memory-at-the-deployment-context)) | 4 | 0 |

The served ceiling stays above window plus summary cap, as official eval does when it resizes a
window (`evaluation_context_overrides`): a summary request carries the whole live history, which
can run past the window. Since arc2-platypus `f380ffd` the Duck's own engine-window derivation
(`_derived_engine_window`, which training turns off) also budgets six times its summary target for
a summary request, the cap of an empty summary's retry with reasoning. Training never takes that
retry (a retained failed draw ends the game), but the served ceiling meets the Duck's bound anyway.
The recipe serves the engine's own window, above both rules (95,232 and 125,952); at 32k both rules
give 49,152, and larger evaluation windows grow (65,536 serves 94,208).

### Resume and segment chaining

- **Resume**: submit again with the same `--run-dir`; the helper reuses the environments the run
  started with. The job reads the run's `frozen_config.json` and frozen source (not `--config` or
  the revision's) and restores adapter, AdamW state and per-rank RNG from the last committed
  iteration; it refuses a changed `training.flash` recipe (the error names each differing field
  with the checkpoint's and the configured value), LoRA identity or world size. Placement is not
  recipe: `device_ids`, `offload_every` and `checkpoint_root` may change, so a run started on the
  shared layout continues under the heterogeneous one (which renumbers the learner's GPUs from 0)
  and back, and so may the diagnostics, the parity thresholds and the publication rounding. No
  publication outlives its job: every resume merges the restored adapter again and broadcasts it
  to the segment's servers before the first window, recording its identity in
  `training_state.json` (`policy_sha256`) and logging whether it equals the one the previous
  segment recorded (the merge is deterministic on the same hardware; a differing identity is
  logged and the new one served).
  Warm starts from other adapter directories are not supported. The length is frozen too: once a run has
  completed its `num_iterations`, a resume logs "nothing to do" and exits, and running it longer
  means editing `num_iterations` in `frozen_config.json` (there is no override).
- **Moving a live run to new code**: the run's `frozen_config.json` is the source of truth on
  resume. `ARC3_FORCE_SNAPSHOT=1` re-freezes `source.tar` from the revision given to the launcher
  but rebuilds the frozen config from the existing frozen config, not from `--config`, and each
  segment derives the servers' `--max-num-seqs` from that same frozen config. To change recipe
  values on a live run: stop it at an iteration boundary (`STOP_AUTO_RELAUNCH` + TERM), edit
  `frozen_config.json` by hand (keep a `.bak`, note the edit in the run's README), and resubmit
  with `--run-dir`. A changed `parallel_rollouts` resizes the engines on that resume. An exported
  `SERVER_MAX_NUM_SEQS` is not re-derived: automatic relaunches carry it, and a resume refuses it
  once it is below the new share (resubmit without it).
  `training.flash.policy_weights_rounding` switches this way too: the resumed segment's
  publication of the restored policy uses the new rounding.
  Run the submit helper from a checkout: its resolver calls `scripts/build_env.sbatch` beside it.
- **Chaining**: segments chain by default (`ARC3_AUTO_RELAUNCH=0` turns it off); the learner stops at the last iteration boundary it can
  reach ten minutes before the limit (exit 75) and the job submits the next segment with the same
  shape and `--dependency=afterany`; `NEXT_TIME_LIMIT` in the run directory sets its `--time`, and a
  `STOP_AUTO_RELAUNCH` file there ends the chain. The learner yields once less than 1.25 times its
  longest recent iteration, plus its checkpoint time and 120 s, remains before that deadline
  (`training/segment_budget.py`). With the recipe's 40-minute games an iteration takes about 41
  minutes (collection sets it; the learner overlaps), so the learner yields between one and two
  iterations before the limit and a 12-hour segment holds about 15 windows (20-minute games: about
  21 minutes per iteration, about 32 windows). Nothing in the launchers assumes the game limit: the collection watchdog (waves x limit
  + 900 s), the overlapped schedule's window estimate and the segment budget all follow
  `max_runtime_minutes`. Each segment is a new job with new servers and a new learner process, so
  a new NCCL group: publications live only in the servers' GPU memory, the run directory keeps the
  adapter checkpoints they are merged from (each with its manifest), and the next segment's resume
  publishes the restored policy to its own servers before the first window. An iteration that overruns its estimate meets Slurm's TERM ten
  minutes before the limit instead (up to a minute earlier); the job takes any TERM
  in the segment's last twelve minutes as that one and chains too, so a manual stop that late needs
  `STOP_AUTO_RELAUNCH`. A segment that does not chain says why (`Not relaunching: ...` in its
  Slurm log). By hand, pass `--dependency afterany:<jobid>` to the helper; the `SBATCH_DEPENDENCY`
  variable is ignored on this cluster.
- A fatal error writes `run_failure.json`; later segments refuse to start until it is removed.
- **Kernel cache**: the learner's compiled kernels (Triton, Inductor) and FLA's autotune picks,
  which FLA stores in the Triton cache, carry over as `kernel_cache.tar.zst` in the run directory;
  a cold first iteration spends 15-25 minutes compiling and autotuning (the warm saving is not yet
  measured). The job unpacks it before the learner starts and repacks it at every exit after the
  learner ran; a failed unpack starts empty and a failed pack keeps the old archive. The node-local
  directory is per run, not per job, because Triton's cache index records absolute paths. Deleting
  the archive resets the cache.

## The model in the learner

| Part | Storage per rank | Training |
|---|---|---|
| Dense weights (attention, GDN, shared expert, norms, embeddings) | BF16, FSDP-sharded per decoder layer | frozen |
| Routed experts | BF16, FSDP-sharded per decoder layer; expanded once from the checkpoint's NVFP4 at load ([Expert storage](#expert-storage)) | frozen |
| Routers | FP32, replicated (`replicate_frozen_routers`) so FSDP gathers stay BF16 | frozen ([trainable](#trainable-routers) with `train_routers`) |
| PLE n-gram table | FP8 E4M3, file-backed on the host; gathered on CPU, scaled to BF16 | frozen |
| Vision tower | BF16, eval mode, `no_grad`, never recomputed | frozen |
| LM head | BF16, replicated; logits upcast to FP32 for log-softmax, loss and entropy | frozen |
| LoRA A/B | FP32 masters, replicated; cast to BF16 in the forward | AdamW (FP32) |
| GDN recurrence | FP32 state inside FLA's chunked kernels; no state between calls | |

A rank holds about 38 GiB after building and peaks near 100 GiB allocated on 35k-token sequences,
about 1.1 MiB more per token of the longest one (133 GiB at 65,536 tokens, 151 GiB at 81,920); the
longest micro-batch sets the peak, and past about 82k tokens a rank needs activation offload
([below](#learner-memory-at-the-deployment-context)). The student must have 48 layers and 10..307
experts, and the servers must serve the same student. Every rank runs every forward: the cycle shared with full-weight training deals
micro-batches longest first, so ranks in one lockstep slot run similar lengths, pads idle ranks with
the cycle's smallest micro-batch at zero loss and normalizes by the global loss denominator, then
`ReplicatedGradientReducer` all-reduces adapter gradients per layer. Rank zero writes adapters and
state; checkpoints keep each rank's RNG.

**Custom kernels.** `FlashSettings` (`training/flash_config.py`) defaults to `kernel_mode: "fast"`
(below): `flex_qsa` attention with the `group_sparse` QSA kernel, FLA's GDN convolution, compiled
PLE convolution and FP32 selector, fused expert permutation, `triton_manual` hyper-connections,
fused norm, gate and packing kernels, and `small_gemm` checkpointing. Against the reference path a
real training cycle runs 2.36x faster (up to 2.5x on the longest sequences), 1.57x from the group QSA
kernel alone; each option's constraints and qualification status are in
[training/flash/README.md](flash/README.md).

### Expert storage

The routed experts are BF16 FSDP parameters like the dense weights: the build expands the
checkpoint's NVFP4 experts to BF16 once at load (`training/flash/reader.py`) and every forward runs
BF16 grouped GEMMs on them. This is not configurable, in either learner. Until 2026-10-09 the RL
learner kept the packed NVFP4 experts instead (`training.flash.expert_storage: "nvfp4"`), a full
copy on every rank, expanded layer by layer in every forward and checkpoint replay. The A/B on the
recipe (job 2923707, 4 B200, kander `flash-bench/tools/ES_REPORT.md`; the live run's adapter 90 on a
28-micro-batch cycle of its probe samples, 672k tokens, longest 32,512) found the two storages
**bitwise identical** (every logprob of a 32.5k-token micro-batch, all 744 gradients and the
post-step weights) and BF16 better on everything but build time:

| Recipe learner, 4 B200 | NVFP4 (retired) | BF16 |
|---|---:|---:|
| optimizer cycle, 28 micro-batches and step (mean of 6) | 46.9 s, 14.3k tokens/s | 29.4 s, 22.9k tokens/s (1.60x) |
| warm 32.5k-token micro-batch, forward+backward / no-grad forward | 7.8 s / 2.9 s | 5.5 s / 1.7 s |
| allocated per rank after build | 42.3 GiB | 38.2 GiB (experts sharded four ways) |
| cycle peak allocated / device peak | 97.9 / 129.2 GiB | 96.6 / 136.6 GiB |
| device peak at 40,960 and 49,152 / 65,536 tokens (of 178) | 129-149 / 164 GiB | 151 / 163 GiB |
| build with warm page cache | 17.5 s | 25.3 s |

BF16 reserves more at 40-49k tokens and still leaves 27 GiB of the device free. By owner decision
BF16 is the only storage, so no configuration can select the strictly worse one; distillation
already defaulted to it ([below](#kernel-mode): 95 and 68 s per warm `fast` update with NVFP4, 70
and 49 s with BF16). Configs and checkpoints written with the retired keys
(`training.flash.expert_storage`; distillation's `expert_storage`, `nvfp4_distribution` and
`model_distribution`) load with one warning and resume, since the experts compute bitwise the same;
a value a key never accepted is refused, and so is distillation's per-GPU replica
(`model_distribution: "replicated"`), which existed only with NVFP4 storage. The single-GPU
route-statistics replay (`training/flash_route_stats.py`, `training/flash_model.py`) is not a
learner and keeps packed experts: the unpruned 512-expert release's BF16 experts (225 GiB) would
not fit one GPU.

### Learner memory at the deployment context

The recipe's segments reach the window plus a reply, 98,304 tokens, which a B200 does not hold
without activation offload. `training.flash.offload_every` N sends every Nth layer's checkpoint
inputs to pinned host memory (`training/flash/offload.py`, torch's `offload_wrapper`) and brings
them back for the backward. Measured on 4 B200s (kander `flash-bench/es98/es98_bench.sbatch`, job
2925049 on dgx01): `flash-bench/tools/es_bench.py` unchanged, the learner as RL builds it from the
recipe's training section with each `offload_every`, the trainer entrypoint's
`expandable_segments`, a fresh adapter; per length one text-only sequence of real token ids, every
rank the same, timed on the second of two slots; the real cycle is 28 micro-batches of the live
run's 8 probe samples (672k tokens, at most 32,512 per micro-batch). Memory per rank, GiB of
178.35; the device peak is nvidia-smi's, sampled each second:

| forward+backward; peak allocated / reserved / device | offload 0 | offload 2 | offload 4 |
|---|---:|---:|---:|
| real cycle (mean of 2) | 29.9 s | 32.6 s | 31.0 s |
| 65,536 tokens | 11.0 s; 133.2 / 140.2 / 148.9 | 83.2 s; 104.4 / 111.6 / 113.6 | 11.5 s; 119.4 / 126.7 / 128.6 |
| 81,920 tokens | 14.2 s; 151.3 / 161.5 / 170.2 | 91.1 s; 115.3 / 125.5 / 127.5 | 14.7 s; 134.1 / 143.8 / 145.7 |
| 98,304 tokens | out of device memory | 93.2 s; 129.3 / 138.9 / 141.6 | **17.9 s; 151.8 / 161.9 / 163.8** |
| 114,688 tokens | not reached | ranks killed: host out of memory | out of device memory |

The recipe offloads every fourth layer: 98,304 tokens leave 14.5 GiB of the device, and the cost
is 3-4% (31.0 against 29.9 s per real cycle, 14.7 against 14.2 s at 81,920). Without offload
81,920 tokens fit with 8.1 GiB left and 98,304 do not. Every second layer holds less on the device
but twice as much on the host, and the job's host memory (4 x 96 GiB, the learner's request in
`scripts/flash_rl_submit.sh`) ran out: from 65,536 tokens the step's resident memory sat at its
384 GiB limit and a micro-batch took 6-8 times as long, and at 114,688 the kernel killed the ranks.
`offload_every` is placement, not recipe, so a resume may change it.

Host memory is the cause: the same benchmark with 240 GiB per GPU (job 2925062 on dgx03, offload 2
and 4 only) ran offload 2 at 16.4 s for 81,920 tokens, 20.6 s for 98,304 and 22.9 s for 114,688
(device peaks 134, 148 and 158 GiB), with the job's memory (page cache included) at 540 GiB for
98,304 and 905 GiB for 114,688. Offload 4 there ran 15.0 and 18.1 s at 81,920 and 98,304, as in
the first job, with the job at 350 GiB, and 22.2 s at 114,688 at a 177.6 GiB device peak, too close
to the card to use. So offload 2 with more host memory for the learner (a larger
`--mem-per-gpu` for its component) would carry segments past 114k tokens at about 15% over offload 4;
the recipe keeps the launcher's 96 GiB and offload 4.

Not covered: real segments carry images (through the frozen vision tower, without gradients) and
several spans; spikes shorter than a second; and the host memory the learner node also gives the
rollout worker's game processes, which the benchmark did not run. A segment somewhere between
98,304 and 114,688 tokens stops fitting at offload 4 (at about 1.1 MiB a token, near 110k); only a
summary request after a long overflow round reaches past the window plus a reply, and one that long
would stop the learner with an out-of-memory error.

### Kernel mode

`training.flash.kernel_mode` switches the learner's whole execution stack between two presets
defined once in `training/flash/kernel_mode.py`, which the distillation (SFT) learner reads too
(`kernel_mode` in its config, [sft/distillation/README.md](../sft/distillation/README.md#kernel-mode)):

| Field | `reference` | `fast` |
|---|---|---|
| attention, QSA kernel | `sdpa` (or `flex_qsa` with `flex`; `physical_sparse` for distillation's single documents) | `flex_qsa`, `group_sparse` |
| GDN / PLE convolution | native / native | `fla` / `compiled` |
| QSA selector | `batched_fp32_top513_v1` | `compiled_fp32_top513_v1` |
| expert permutation | `weighted_permutation` | `fused_permutation` |
| hyper-connections, fusions | none (fused norm tail only) | `triton_manual`; GDN norm, MLP gates, GDN QKV, QSA norm |
| checkpointing | `full` | `small_gemm` |
| frozen routers | FSDP-sharded | replicated |
| Blackwell GDN guard | off (distillation's unpacked inputs only) | on |

`reference` is the stack the cached-teacher distillation results were produced with: the
builders' own defaults (`build_flash_model`, `sft.distillation.flash_model.build_model`). `fast` is
this learner's stack as it was before the switch existed, so the shipped recipe trains exactly as
before; `tests/test_flash_kernel_mode.py` pins both. RL defaults to `fast`, distillation to
`reference`. Any field set beside `kernel_mode` overrides that one choice of the preset (for
experiments); the loaded settings, `effective_config.json` and the trainer log ("Flash kernel
stack: ...") hold the resolved stack. The label and every resolved field are part of the resume
identity: a checkpoint resumes only under the stack it was trained with, and states saved before the
switch read as `fast`.

The learner packs documents, so whatever the mode it keeps the GDN guard (the eight-warp output
configuration is wrong on packed inputs) and uses Flex rather than the single-document
`physical_sparse` kernel under `reference` with `flex_qsa`. The switch covers model execution only.
Placement stays each pipeline's own: CPU offload (`offload_every`; experts are BF16 in both, see
[Expert storage](#expert-storage)). So do the cycle's length-aware dealing across ranks, the sharded
probe and trust-region diagnostics and the job's kernel-cache archive, in both modes (they change
scheduling and summation order, not the model). Rollout-side settings are untouched.

Measured on the published e276 router-distillation recipe (e307 teacher, 4 B200, the SFT entry
point; `flex_qsa` with selection reuse and retention, `offload_every` 3, GA8; jobs 2919562,
2919567, 2919827-2919829, 2919933). With identical weights on the 64 held-out probe rows (62k
completion tokens), `fast` moves the student's per-token distribution by KL 0.018 nats/token
(p95 0.086; |delta log p(target)| mean 0.071; 95.8% top-1 agreement) from `reference`, as far
as `reference` moves between its two historical attention settings (SDPA against FlexQSA: 0.017,
p95 0.081, 95.8%); the probe's KL to the teacher is 0.04276 against 0.04224 (SDPA 0.04262). On the
same rows the first update's loss per token differs by 1.3% (SDPA: 1.4%) and the router gradient
norm by 0.03% (SDPA: 1.1%). After two Adam updates, whose first steps are nearly sign(g), the router
update agrees with reference's at cosine 0.51 (SDPA 0.55). Each stack is deterministic: a rerun in a
new process reproduces the routers bitwise. Warm updates (kernel caches from a previous run of the
same rows) took 70 and 49 s under `fast` against 144 and 145 s under `reference` (2.1-3.0x); with
the since-retired NVFP4 expert storage `fast` took 95 and 68 s. Cold updates took 3-6x as long as
warm ones under `fast` (2x under `reference`), so the distillation job keeps a `fast` run's kernel
cache across segments.

**The Blackwell GDN guard.** `training/blackwell_gdn.py` (installed by `PPOTrainer.__init__` before
any forward on B200, FLA 0.5.1, Triton 3.6.0 or 3.7.1) keeps
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` to its two-warp configurations (FLA #945 / Triton
#9871 report a four-warp race) and removes the eight-warp BK=BV=128 configuration of
`chunk_fwd_kernel_o`, which the autotuner picks and which returns nondeterministic, wrong outputs on
the learner's packed (`cu_seqlens`) path (11-41% relative error from token 32 of each chunk): two
identical learner forwards differed by 0.27-0.31 nats per sampled token. With the guard the learner
is bitwise repeatable, and a sweep of every autotuned FLA kernel it runs found no other faulty
configuration. FLA persists autotune picks across processes in `TRITON_CACHE_DIR`, so the job gives
the learner a per-run Triton cache; probes and tests need their own as well.

**Images.** With `training.multimodal = true` the vision tower (`FlashMultimodalDecoder`) runs
before the checkpointed decoder layers. Collation (`prepare_training_media`) reproduces the
witnessed pixels, grids, token types and `[3, B, T]` M-RoPE positions; image features replace the
embeddings at `image_token_id` placeholders while the PLE still sees the original ids, and text-only
batches take the same route so ranks issue identical FSDP collectives. Limits: still images,
complete cache-free documents, right padding; truncated or mismatched media evidence is refused.
Servers verify each image request's evidence and must not run `language_model_only`. Images add
nothing measurable to the parity floor.

## LoRA

`LoRALinear` (`training/flash/dense_lora.py`) wraps the targets in `training/flash/lora_targets.py`:

| Module | Layers | Targets |
|---|---:|---|
| QSA attention (`self_attn`) | 12 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| GDN (`linear_attn`) | 36 | `in_proj_qkv`, `in_proj_z`, `out_proj`; optional `in_proj_b`, `in_proj_a` |
| Shared expert (`mlp.shared_expert`) | 48 | `gate_proj`, `up_proj`, `down_proj` |

Routed experts, routers, embeddings, the PLE table, the vision tower and the LM head carry no
adapters. `training.lora.target_modules` must be explicit; unknown, duplicate or unmatched names
fail. The recipe uses all twelve at r16/alpha 32: 33.5M parameters in 744 tensors (67.0M at r32;
30.5M and 60.9M without `in_proj_b`/`in_proj_a`). `training.lora.dropout` must be 0 (there is no
dropout path; the 0.05 default is rejected). B starts at zero, so the first policy is the base
model.

**Publication: merged weights, not adapters.** After each iteration rank zero writes the adapter
(`checkpoints/adapter_iter_NNNNNN/`, FP32, PEFT 0.18.1 format): it is the training state that
resume reads, and the durable record of the policy; it is never served. What the servers receive is
the policy itself (`training/flash_policy_weights.py`): every publish, the initial one included,
every rank gathers each target's FSDP-sharded BF16 base weight in one order, and rank zero merges it
with the replicated FP32 masters, `W + alpha/r * B @ A` computed in FP32 (TF32 off) and rounded once
to BF16 with a fixed dither (below), and takes each tensor's checksum on the device
(`policy_checksum`: its BF16 bit patterns weighted by their position in HF row-major order,
[below](#merged-policy-weights)). The merged tensors stay on rank zero's GPU; nothing is written
but the publication's manifest, `adapter_iter_NNNNNN/policy_weights.json` (format
`flash_policy_weights/3`: the tensors' HF names (`model.language_model.layers.{i}.<module>.weight`)
in layer-then-name order, their shapes and checksums, the iteration, LoRA targets, whether routers
are included, the merge's rounding, and the publication's identity `policy_sha256`, the sha256 of
the manifest's canonical JSON). The trainer state records the same identity. At the next rollout
boundary rank zero broadcasts the tensors in the manifest's order over NCCL straight into every
server's GPU memory ([Merged policy weights](#merged-policy-weights)) and frees them; no publication
bytes touch disk, so there is nothing to push, prune or point `latest_runtime_adapter` at. For the
recipe that is 372 tensors and 5.44 GiB per publication; the initial policy (B = 0) publishes the
checkpoint's bytes.

**Why the rounding is dithered.** Rounding to nearest returns the base weight wherever an element's
update is below half a BF16 ulp of it, which is most elements: at the live run's iteration 50 a
nearest-rounded merge changed 10-31% of them and carried 69% of the adapter's per-token logprob
effect (slope 0.69 against an engine serving the adapter as LoRA, the learner's own forward on
those weights 0.68), sitting 0.064 nats per token from the learner's LoRA forward against a 0.04
floor; earlier in a run the servers would serve nearly the base model. The merge therefore adds 16
uniform random bits below the BF16 mantissa of the FP32 sum and truncates: each element rounds away
from zero with probability equal to its position between its two BF16 neighbours, so the update
survives in expectation (the same diagnostic: slope 0.985, 0.041 from the LoRA forward). The bits
are drawn once per tensor from a seed derived from its name (`dither_seed`), so publications are
deterministic, an element whose sum does not move keeps its value from one publication to the next,
and B = 0 still publishes the checkpoint's bytes. Routers are not merged: they ship as the
nearest-rounded casts the learner routes with. The dither is the default of
`training.flash.policy_weights_rounding`; `"nearest"` publishes the plain BF16 cast of the same FP32
sum instead, to compare the two in evaluation, and every publication records which it used
(`rounding` in `policy_weights.json`, hence in its identity). It shapes what the servers sample,
not what the learner computes, so it stays out of the resume identity, and every resume republishes
the restored policy (a collective runtime command) with the configured rounding, so a segment can
switch it. Measured cost per publish of the merge on the recipe's four ranks (MW): 3.2 s gathers,
merges and checksums (every rank waits that long); the file write that followed (11.3 s on rank
zero) is gone, replaced by the broadcast (below). Publishing the same policy twice gives the same
bytes, so the same identity.

**Other setups need code.** The shared stack has routed-expert LoRA (`expert_lora.py`), but
`FlashPPOTrainer` builds only `dense_lora_policy` and the publication knows only the dense targets
and routers (the servers refuse any other name). Routed experts stay frozen: training them would
mean shipping and requantizing the NVFP4 experts every iteration.

### Trainable routers

`training.flash.train_routers = true` trains the 48 routers (307 x 2560 each, 37.7M parameters)
beside the LoRA. It is off by default: **the shipped recipe keeps routers frozen**, and no
experiment has run with it.

- **Learner.** The routers are the `routers` parameter group of the dense-LoRA trainability
  (`TrainabilityPolicy.train_routers`): FP32 masters replicated outside FSDP like the LoRA masters,
  so `replicate_frozen_routers` (default on) finds no frozen router and the two compose; their
  gradients reduce with their layer's LoRA bucket in `ReplicatedGradientReducer`. AdamW gives them
  `training.flash.group_learning_rates.routers`, by default the base learning rate under the same
  schedule, and no weight decay: decay pulls toward zero, which for a router is uniform routing,
  not the base model. Their gradients count in the global gradient norm and its `max_grad_norm`
  clip, and in the trainable-parameter norms. The learner routes with
  `F.linear(x, weight.to(x.dtype))`, so a master's BF16 cast is exactly what it uses. The masters
  ride in the trainer state (`flash_router_masters`) and resume restores them exactly;
  `train_routers` is part of the Flash recipe identity, so a resume cannot flip it (states from
  before the option resume as false). Refused outside `weight_mode = "flash"` and with
  `reference_kl_coef > 0`, whose disabled-adapter reference forward would carry the trained
  routers. Memory: master, gradient and two moments, about 600 MB per rank.
- **Publication.** Every publish, the initial one included (the checkpoint's routers unchanged),
  adds the routers' BF16 casts to the merged policy weights under their HF names
  (`model.language_model.layers.{i}.mlp.gate.weight`, 75.4 MB), declared in the manifest
  (`routers: true`); servers receive them with everything else ([Serving](#merged-policy-weights)).
  The adapter directory keeps its own copy, `routers.safetensors` and `routers.json` (`sha256`,
  `iteration`, `tensors`), written canonically by `training/flash_routers.py`, for loading an
  adapter back (a resume reads the adapter directory; the publication itself lives only in the
  servers' memory); the PEFT file excludes them. The router file is written and read by one
  canonical BF16 safetensors writer and strict reader, `training/flash_safetensors.py`. Without
  `train_routers` no router is published and the servers keep the checkpoint's.
- **Serving.** The fork's MoE runner reads the router's `ReplicatedLinear` weight on every forward,
  so the in-place update reaches every token; only its fused shared-expert gate (ROCm AITER) keeps a
  private copy, and an engine on that path refuses policy weights.
- **Parity gate.** The lag-aware check snapshots every trainable parameter
  (`_remember_parity_policy`), so a lag-one check scores with the sampler's routers as well as its
  LoRA and restores the current ones afterwards; no files are read. A resumed run needs no older
  routers, because its first window is collected with the restored policy.
- **Load balance is observed, not optimized.** There is no balancing loss; the RL loss alone moves
  the routers. Their load balance is the [expert load](#training-signal-details-that-matter)
  metrics' (`train/`, `rollout/` and `eval/expert_load_*`), which count the trainable routers'
  decisions like the frozen ones'. Their own metric is drift: each window reports
  `train/policy_update/router_relative_drift` (and a `routers:` log line), the Frobenius norm of
  the masters' change from the checkpoint's routers over the checkpoint's norm, all 48 layers
  together.
- **Measured** (R1 smoke on kander, `flash-bench/tools/R1_REPORT.md`). On the recipe's learner (four
  B200s, eight real micro-batches, one step at the first warmup learning rate, 1.25e-7) the reduced
  router gradient norm was 0.22 (every layer non-zero; LoRA 0.15), and gradients and FP32 masters
  hashed identically on all ranks. That one step (largest element change 2.4e-7) moved the base
  model's logprobs on a fixed text by 0.040 nats per token: a few BF16 casts flip by one ulp and
  routing flips amplify them, so router updates show at the parity floor's scale from the first
  step. Trained routers add 450 MB to each trainer state (721 MB). Routers perturbed by one standard
  deviation of noise (learner effect 0.69 nats per token) moved the served logprobs by 0.67-0.69,
  with the server 0.21-0.24 from the learner (0.69 had the update been ignored); an engine started
  from a checkpoint carrying those routers sat 0.23-0.24 from the learner and 0.21 from the updated
  engine, which is the engine's own distance from itself on a repeated identical request (0.18-0.22;
  0.10 with the checkpoint's routers, where server and learner differ by 0.11). An update is
  indistinguishable from loading the routers at startup: the same in-place mechanism the merged
  policy weights now use for every tensor. (R1 measured the router-only file and route that the
  merged publication replaced: 0.7 s and 75.5 MB per publish, 0.31 s per server apply.)

## Serving

Each engine is `scripts/launch_platypus_server.sh` with `SERVER_WEIGHT_MODE=flash`, running
`training.native_vllm_server` from the harness environment on the fork its lock pins,
`Tufalabs/arc-prize-2026-vllm@3bda1ba09` (`0.29.1.dev166+g3bda1ba09`). `SERVER_MODEL` is the local
checkpoint directory (no HF revision); the job sets each server's GPU, port, length and image mode.

The table gives each flag and each setting of the harness's Flash deployment that the launcher does
not mirror, with the reason: ullm's `deployment.py` at `e53d822d` (`_flashnext_config`, `build()`
and its Flash `slurm-b200` cell) and what the ARC-2 Duck's `presets.configure` adds for its Flash
column at `f380ffd`. `tests/test_flash_harness_serving.py` reads both from the submodules and fails
on a harness setting, or a changed harness value, that is neither mirrored nor stated here.

| Flag | Why |
|---|---|
| `--kv-cache-dtype auto` (BF16 KV) | parity: BF16 KV lowers the engine's own floor (mean 0.055 against 0.062 with FP8 KV), KV is not the limit (peak use about 11% of the pool), and FP8 KV measured no speed gain. `SERVER_KV_CACHE_DTYPE=fp8_per_token_head` is the fallback |
| no `attention_config.indexer_kv_dtype`: the fork's `auto`, BF16 cached keys for the Qwen4Exp sparse-attention indexer (`vllm/models/qwen4_exp/nvidia/indexer_qsa.py` at `3bda1ba09`) | closer to the learner, which computes the indexer in BF16/FP32: the cached keys choose the blocks each query attends to, so their precision is part of the served distribution. ullm's Flash engine keeps BF16 too; the ARC-2 column sets `"fp8"` (`presets.py`, its Flash engine layer: "+5.9% generated and +7.1% pool on the four L4s' traffic replay, and no score cost in sweeps 8 and 9"), a throughput choice |
| `--kernel-config '{"moe_backend":"marlin"}'` | Marlin NVFP4 MoE: the harness's backend and the one every parity measurement used |
| `--mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode align` | an FP32 recurrent state gave no parity gain and costs 15% of KV capacity; `align` is the mode compatible with prefix caching |
| `--engram-config '{"cpu_offload": true}'` | only the frozen 47.7 GB PLE table sits in pinned host memory (allocated there, rows read over UVA), freeing GPU memory for KV; the harness deployment's setting. It is a weight, not KV: the launcher refuses any other CPU or KV offload (and `--cpu-offload-params`, whose generic offload would build the table on the GPU first), and the server attests that Engram offload is the only one |
| `--no-enable-flashinfer-autotune`; `--compilation-config` (`triton.autotune_at_compile_time: false`) | FlashInfer autotuning off, as the harness. The compilation setting is inert at this fork: Qwen4Exp runs without `torch.compile` and with breakable CUDA graphs (upstream #55272); it stays for a pin that compiles again, where compile-time tuning cloned the host PLE table onto the GPU |
| `--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192` | episodes resend a growing prefix (88% cache hits); long prompts prefill in chunks. 16,384 measured no gain |
| `--max-num-seqs $SERVER_MAX_NUM_SEQS` | the job derives `ceil(parallel_rollouts / servers) + 8` ([Sizing](#sizing-and-scheduling)): 40 for the recipe (128 attempts over four servers, 32 in flight on each); the launcher has no default. The ARC-2 column runs 230 concurrent games per engine (its `max_concurrent`, which the Duck turns into `max_num_seqs`) over a host KV tier |
| `--gpu-memory-utilization 0.95`, the memory profiler's CUDA-graph estimate kept | the harness boots with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` and its budget lowered by that estimate (ullm 0.9488; the ARC-2 column 0.9174 at 230 sequences): the same pool, with one graph capture instead of two. That saves boot time only (the estimate's capture took 40 s drafted and 15 s undrafted on one B200), and the equivalent budget moves with drafting and the sequence count (0.9436 drafted at 40 sequences, kander job 2924583; 0.9481 undrafted, job 2925377), so one fixed figure would shrink the pool for one shape or eat the capture's headroom for another |
| `--async-scheduling --watermark 0.10` | the harness's (ullm `build()`), mirrored: the next step is scheduled while the current one runs (the fork enables this when unset; stated, an incompatible setting fails at startup instead of being dropped with a warning), and 10% of the KV pool stays free when a waiting request is admitted beside running ones, so a full pool queues admissions instead of preempting decodes. Neither changes what a request samples; the watermark is inert below a nearly full pool, far above RL's peak |
| `SERVER_MODEL` the recipe's `training.flash.checkpoint_root` (e307); drafting with ullm's e307 drafter | ullm's Flash cell serves this checkpoint and drafter (`FLASHNEXT_E307_CHECKPOINT`, `FLASHNEXT_E307_DRAFTER`). The ARC-2 column serves our REAP-307 prune and its drafter instead (`presets.py`: `E307_ARC2_CHECKPOINT`, `E307_ARC2_DRAFTER`); RL trains e307, the model of every parity and drafting measurement here |
| no `--enable-lora` | the policy arrives as merged weights ([below](#merged-policy-weights)); LoRA costs 20-26% of decode and 23-30% of prefill and pins Marlin MoE and no MTP |
| `--generation-config vllm --override-generation-config '{}' --logprobs-mode processed_logprobs` | no checkpoint sampling defaults; logprobs include admitted processing such as the summary bias the learner replays |
| `--summary-forbidden-token-ids 248058` | the one id a request may bias, the summary's ban on `<tool_call>` (the recipe's `agent_summary_forbidden_token_ids`, the Duck's `QWEN_TOOL_CALL_TOKEN_ID`): the server refuses to start unless it decodes to `<tool_call>` under the served tokenizer, and refuses a bias on any other id ([docs/native_serving.md](../docs/native_serving.md)) |
| `--limit-mm-per-prompt '{"video":0}' --mm-encoder-attn-backend FLASH_ATTN --mm-processor-cache-gb 0 --skip-mm-profiling` | images for ARC-2 and ARC-3 image modes (`'{"image":0,"video":0}'` otherwise). The server processes each image itself and hands the engine the verified tensors, with the processor cache off, so the harness's `mm_processor_cache_type="shm"` (its fix for the per-process `lru` mirror's eviction crash, vllm#31404) has no cache to apply to |
| no `max_pixels` (no `--mm-processor-kwargs`) | the image processor shrinks only above its checkpoint's ceiling, 16,777,216 pixels (`preprocessor_config.json`), orders of magnitude above any image the Duck renders. The deployed Duck sets `max_pixels` to its own largest render instead (`derive_max_pixels`, `duck/images.py` `largest_image_pixels`); training turns that derivation off because the engine is the server's (`EngineProxyFactory` has no engine arguments), and neither ceiling shrinks an image, so both process every image alike. Each server states its ceiling as `image_max_pixels` on `/metadata` ([docs/native_serving.md](../docs/native_serving.md)) for the worker's window-start check that the Duck's largest render fits. A mirrored `max_pixels` would also have to enter the trainer's processing identity (`mm_processor_kwargs={}`, `training/multimodal.py`) |
| `--dtype bfloat16`, `VLLM_USE_V2_MODEL_RUNNER=1` | BF16 dense weights (experts are NVFP4 by checkpoint, which the fork reads from its `quantization_config`, as ullm's explicit `modelopt_fp4` names); the fork's runner for this model |
| no host KV tier (ullm's `cpu_kv_offload`; 96 GiB on the ARC-2 column), no `embed_tokens` offload (`cpu_offload_params`) | KV stays on the GPU by owner decision (2026-09-04), and the pool is not the limit (peak about 11%), so neither the tier nor the 1.18 GiB the host-resident input embedding would free buys anything; `/runtime` attests that the Engram PLE table is the only host-resident weight |
| no `prefix_match_unit`, `prefix_cache_decode_tail` or `prefix_cache_context_threads` (fork defaults: block-granular prefix keys, no decode tail, no threads) | the harness's park mode (ullm: 80, on, on) keeps each conversation live between turns over its host tier, which RL does not run. The unit must divide the attention block: 80 divides the harness's FP8-KV block at four drafts (1,840 tokens) but not the recipe's (BF16 KV, drafted: 928), so its engines would refuse to start. Threads also need the client's per-request tag, which the gateway strips ([docs/native_serving.md](../docs/native_serving.md)) |
| no `--prefix-cache-retention-interval` | with drafting, as the recipe drafts, the fork itself keeps a recurrent state at every block boundary (`None`, ullm's explicit setting): its default for a hybrid model under EAGLE-family drafting, since its sparse default could never hit there (`vllm/engine/arg_utils.py` at `3bda1ba09`; a launch for the recipe's shape resolved `None`, kander job 2925454). Undrafted servers keep the sparse default (0, semantic checkpoints only), at which the 88% hit rate and the undrafted parity floor were measured; changing it moves which prompt tokens a continuation recomputes |
| vLLM's CUDA-graph ladder (no `cudagraph_capture_sizes`) | ullm's `dense_cuda_capture` (a factory setting) captures a graph for every decode width up to `max_num_seqs`, so a drafted decode batch between the ladder's rungs is not padded up. More graphs cost capture memory outside the 0.95 budget and boot time, and a graph's width changes the batch shape the kernels see; a decode-throughput lever to measure first |
| no `--reasoning-parser` | ullm's engine names `qwen3` only to fill the engine's reasoning config, which enforces `thinking_token_budget`; the native server parses nothing (the worker's ullm backend splits reasoning) and the gateway refuses `thinking_token_budget` |

GDN prefill is left to the fork's automatic choice (FlashInfer on B200), which matches an
independent server at the floor. With BF16 KV the fork falls back from FA4 to FA2 for attention (FA4
rejects head size 256 with the 800-token blocks the hybrid cache forces; still so at `3bda1ba09`);
cost unmeasured. Since `3bda1ba09` the fork runs Qwen4Exp eagerly with breakable CUDA graphs
(upstream #55272: no Inductor fusion of the glue around the kernels) and fuses the PLE residual add
and the QSA output gate into their kernels (#55309); on one B200 this measured within 0-3% of
`077920f3` in prefill and decode, and its logprobs sit at the same floor against the learner
([Sharp edges](#sharp-edges-of-merged-publication)).

The harness's own deployment of the checkpoint shares the stability settings above (Marlin,
FlashInfer autotuning off, the PLE table on the host through Engram, BF16 recurrent state in `align`
mode, 8,192-token chunks) and, like RL now, serves without LoRA and drafts; the table gives each
setting it runs that RL does not. Serving merged weights removed the LoRA gap (an active r16 adapter cost 20-26% of
decode and 23-30% of prefill throughput; measured against merged weights below) and frees the
engine for native FP4 MoE (1.4x decode without LoRA, not yet qualified) and MTP drafting (an
opt-in flag, [below](#speculative-decoding-opt-in)), so the gap that matters now is slot count. Concurrency was the main unused lever: at 16 attempts
per server a server kept 0.69 of its slots busy on average (finished attempts wait for the window's
stragglers), KV peaked near 11%, and decode throughput kept rising up to 64 concurrent requests.
Hence the recipe's 32 attempts per server, twice the old load, before more nodes
([Sizing](#sizing-and-scheduling) says why not 64); at that load KV should reach roughly twice the
old peak (unmeasured). Refilling slots as attempts finish (a window-semantics change) is the next
lever.

### Speculative decoding (opt-in)

`inference.speculative_decoding` drafts with e307's MTP head on every Flash server. A
config without it does not draft; the shipped recipe turns it on. `{"enabled": true}` fills in
ullm's e307 drafting (its Flash deployment, `deployment.py` at `e53d822d`, copied in
`training/flash_drafting.py`; a test reads it back from the pinned submodule; the ARC-2 Duck's
column drafts with REAP-307's own head, [Serving](#serving)): the drafter `FLASHNEXT_E307_DRAFTER`
(`/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform-drafter-v1-nvfp4-dv16k`,
e307's MTP head fine-tuned on ARC play, NVFP4 experts, a 16,384-id draft vocabulary) and four
draft tokens; `drafter` and `num_speculative_tokens` (1-8) can be set. Any other weight mode refuses
it. `training.job_shape` hands each server the fork's `--speculative-config` as
`SERVER_SPECULATIVE_CONFIG` (`flash_rl.sbatch`, `flash_rollout_node.sh`):

```json
{"method": "mtp", "model": "<drafter>", "num_speculative_tokens": 4, "quantization": "modelopt_fp4",
 "draft_sample_method": "probabilistic", "gdn_recoverssm": true, "disable_eagle_block_drop": true}
```

The launcher adds it in flash mode only and refuses `--speculative-config` in `SERVER_EXTRA_ARGS`.
The drafter is read from shared storage, not copied to node scratch. The setting is in the
inference section, so the frozen config keeps it through segments and resumes; it is not part of
the Flash recipe identity, so a resume may switch it.

**The sampled distribution does not change.** Drafts are sampled from the drafter and verified by
standard rejection sampling against the target's processed distribution, so every emitted token
(accepted draft, the token recovered after a rejection, the bonus token) is a draw from the
policy. Its logprob is the target's at that position: at `3bda1ba09` the V2 runner's
`RejectionSampler` computes the logprob (and the top-k for `score_centering_top_k`) of each sampled
token from the processed target logits it verified with, and the scheduler hands each request one
row per emitted token. The gateway's checks (one sampled-token logprob per token, the score-centering
head per token) are unchanged. The drafter changes only speed. It is never trained and was tuned
on e307, so acceptance falls as the policy drifts; that is accepted.

**What the servers attest.** `/runtime` adds `speculative`: the engine's resolved drafter path,
method, draft count, draft sampling, rejection-sampling method and both acceptance thresholds,
draft top-k/top-p, RecoverSSM, block-drop and drafter quantization. A server serves only the
harness's drafting with exact verification (standard rejection sampling, thresholds 1.0, sampled
drafts, NVFP4 drafter, RecoverSSM) on Qwen4Exp; the fork's relaxed or synthetic acceptance, lossy at
temperature 1, is refused. The trainer's `ensure_server` requires exactly the drafter and draft
count its config names and refuses a drafting server when the config asks for none, and the
converse, as for the Engram offload.

**Publications and the GDN guard.** The drafter is a separate module: one full-attention decoder
layer with its own LM head, no GDN layer and no LoRA target. A merged-weights publication touches the
target model only (`get_model()`; the census is the target checkpoint's safetensors), so the write
audit, the 372 checksums and the address check cover the same parameters as without a drafter, and
the drafter keeps its e307 weights while reading the new policy's hidden states. The Blackwell GDN
guard is process-wide and installed before either model loads; it constrains the prefill
chunk-state autotuner. In a drafted decode step the target's GDN layers verify through RecoverSSM's
kernels, which have fixed launch shapes (no autotuner).

Measured on one B200 (kander `flash-bench/tools/sd_server.sbatch`, job 2924583, the HP smoke's
inputs: the live run's checkpoint-50 publication, MW's 16 real ARC-2 prompts of 5-30k tokens and
their 3,428 learner-scored tokens, 49,152 context, 40 sequences, BF16 KV; the launcher with the
default drafting). The later adapter is the live run's iteration 90, published on the same GPU with
the learner's merge rule (the same script reproduced the iteration-50 publication byte for byte).
Mean |delta logprob| per token (p95), against the same comparisons without drafting (HP smoke):

| | drafting | without |
|---|---:|---:|
| server prefill vs learner, published weights (iteration 50) | 0.0353 (0.168) | 0.0342 (0.164) |
| server's decode vs its own prefill, iteration 50 | 0.0370 (0.165) | 0.0326 |
| the same server's prefill vs a repeated prefill, iteration 50 | 0.0376 (0.166) | 0.0324 |
| decode vs prefill of draws through the gateway (`NativeService.generate`), iteration 50 | 0.0330 (0.159) | |
| server's decode vs its own prefill, iteration 90 | 0.0297 (0.130) | |
| server's decode vs its own prefill, e307 weights | 0.0705 (0.264) | not measured |
| prefill, drafting vs undrafted server: iteration 50 / e307 weights | 0.0336 / 0.0394 | |

At the published weights the floors hold. On e307's own weights the decode-prefill distance is
twice that, with no undrafted reference measured; the two servers' prefills of the same tokens also
differ more there (0.039 against 0.034), which points at those weights rather than the verify path,
but this is not established. Gateway draws carry one sampled-token logprob per token (16 of 16
draws); with `logprobs=32` every token carries a 32-entry head, and where the sampled token is in it
(1,788 of 1,825) its logprob equals the head's exactly.

| one B200, tokens/s | without (HP) | e307 weights | iteration 50 | iteration 90 |
|---|---:|---:|---:|---:|
| prefill, 16 / 32 requests | 23,718 / 23,640 | 23,063 / 23,038 | 23,203 / 23,173 | 23,197 / 23,156 |
| decode, 16 requests | 1,241 | 1,931 | 2,024 (1.63x) | 2,044 (1.65x) |
| decode, 32 requests | 2,104 | 2,845 | 3,172 (1.51x) | 3,121 (1.48x) |
| draft acceptance (tokens per verify step) | | 49-50% (2.94-2.98) | 55-61% (3.21-3.44) | 56-61% (3.23-3.43) |

Acceptance did not fall by iteration 90: it is higher on the trained policy than on e307 (per draft
position at 32 requests: 75/54/39/29% on e307, 80/62/49/39% at iteration 50). The KV pool is
3,967,385 tokens (4,319,837 without, -8.2%; 114.93 against 117.84 GiB, attention blocks of 928
tokens instead of 800). Startup took 233 s to ready (163 s without). The trainer's sync of the
iteration-50 publication applied all 372 tensors with every checksum equal, `load_weights` 2.29 s
and checksums 0.04 s (2.47 and 0.04 without; its 52 s read was a cold /data read on that node),
pinned the warm-up identity and passed the window-start check; a publication with a flipped byte
was refused (HTTP 503, engine poisoned). The trainer's `ensure_server` accepted the server with the
config's drafter and refused it for a config without drafting, another drafter path and three
drafts.

For the 4:4 split at 128 attempts (LIVE_REPORT run C): collection 983 s per window, the learner's
pass about 0.49 of it (482 s), which the BF16 expert storage would cut 1.60x (301 s). Drafting
speeds decode about 1.5x at 32 requests and leaves prefill unchanged: if a window then collects
1.5x the tokens in the same 983 s (that run's 900-s game limit sets its length; the recipe's is now
1,200 s), the learner needs 723 s
(NVFP4) or 452 s (BF16); if it collects the same tokens in 655 s, 482 or 301 s. Rollout stays the
bottleneck either way; the learner would become it only past a 2.0x (NVFP4) or 3.3x (BF16) rollout
speed-up.

### Merged policy weights

A publication goes over the servers' NCCL weight routes (`/weights/*`, the routes the 27B
`full_weight` mode uses; [docs/native_serving.md](../docs/native_serving.md)). Flash servers start
with `--weight-transfer-config '{"backend":"nccl"}'` and no LoRA. Once per learner process (at the
first publication) `/weights/init` on every server, run concurrently with the learner, forms one
group: the learner's rank 0 is NCCL rank 0 and each server's worker takes the next rank, at the
rendezvous of the learner node's name and a free port. The learner does not contain vLLM, so its
side of vLLM's protocol (a TCPStore on that port, NCCL's unique id in the store as the
`ncclUniqueId` vLLM's workers unpickle, `ncclCommInitRank`, the communicator's warm-up all-reduce)
is `training/policy_sync/stateless_nccl.py`, which calls the `libnccl` torch loaded (torch 2.13 and
NCCL 2.29.7 on both sides) through ctypes. A server that restarts is fatal (its instance id no
longer matches), so the group is never rebuilt within a segment; each segment builds its own.

Each publication runs at a rollout boundary, with every server drained (`activate_policy_at_boundary`,
the full-weight sequence; the learner waits for the window in flight): `/weights/pause`, then
`/weights/start` with the label (`kind: "policy_weights"`, `policy_sha256`, `policy_iteration`) and
the manifest. The server checks that the label names the manifest's sha256 and iteration, and each
vLLM worker (`BlackwellGdnWorker.prepare_flash_policy_weights`, called by name through
`collective_rpc`) checks the manifest before any byte moves: exactly the checkpoint's tensors for
the declared LoRA targets (and routers when declared), read from the checkpoint's safetensors
headers, with the checkpoint's shapes and BF16 dtype, and the same census as the engine's first
publication. It then fixes where each tensor belongs without asking the loader
(`PolicyWeightReceiver.homes`): a leaf vLLM packs (`qkv_proj` = q, k, v; `gate_up_proj` = gate, up;
`in_proj_qkvz` = in_proj_qkv, in_proj_z; `in_proj_ba` = in_proj_b, in_proj_a, the fork's
`packed_modules_mapping` order) belongs at its parts' running row offset in
`layers.{i}.<parent>.<packed>.weight`, whose rows the parts' checkpoint shapes must tile exactly; any
other tensor is the whole of `layers.{i}.<parent>.<leaf>.weight`; each home must be a contiguous BF16
matrix of that shape, or the publication is refused. `/weights/update` then receives: rank 0
broadcasts each tensor in manifest order, and each worker
(`receive_flash_policy_weights`) takes it into a buffer of the declared shape on the group vLLM
opened and loads it through the model's own `load_weights` alone, the startup path with its HF to
vLLM mapping and packed modules, recording every copy into a served parameter (any other in-place
write raises) and requiring every copy of a tensor to land inside its home's rows; homes never share
rows. A refused load still receives (and drops) the remaining tensors, so the broadcast completes
and the refusal comes back as the update's error rather than a hung collective. vLLM's own
`update_weights` is not used: the fork's NCCL engine reloads layerwise, moving every layer to the
meta device and back and materializing a partly loaded layer uninitialized before copying it whole,
which a publication of some of a layer's tensors cannot pass through. `/weights/finish` then runs
`verify_flash_policy_weights`: every tensor's checksum recomputed from its home's rows, read in HF
order, must equal the manifest's, and no parameter may have moved (CUDA graphs keep reading the same
storage); the server resets the prefix cache (cached K/V and recurrent states belong to the old
weights). `/weights/resume` makes it live: from then on the base model's identity carries
`kind: "policy_weights"`, `policy_sha256` and `policy_iteration`, so every draw names the
publication that served it. Any refusal or mismatch poisons the engine: a server that fails a
publication never serves again, and the learner stops. `GET /policy_weights/info` returns the
published policy's `sha256` and `iteration` (nulls while the checkpoint's weights are served).
Adapter loads are refused on these engines; the 27B `lora` mode keeps its adapter endpoints.

`policy_sha256` is the sha256 of the manifest's canonical JSON, checksums included. Each checksum
pins its tensor's elements in place (below), so the identity pins the served bytes as the file hash
did when publications were files, and no publication bytes touch disk. After the resume the trainer
checks each server's resumed identity against the label (`RolloutCollector.confirm_policy_weights`),
then `/policy_weights/info`, then a one-token warm-up under the base served name, which must carry
the same identity on the same server instance. That identity is pinned for the window, and the
rollout gateway admits only draws carrying it. Every window start checks `/policy_weights/info`
again; another hash or a restarted instance (back on the checkpoint's weights) is fatal.

Why no probe is needed: the server returns, per tensor, a checksum of the logical matrix it serves
in that tensor's place, computed after every copy, and the trainer compares it with what the learner
merged, so a dropped, stale, partial, mis-scaled, mis-placed or permuted publication fails the
publication instead of hiding under the parity floor. The checksum (`policy_checksum`) is
`sum_i bits_i * ((i mod 65521) + 1)` over the matrix's row-major HF order, `bits_i` each BF16 bit
pattern as a signed int16: moving an element to another weight changes it, and two positions fewer
than 65,521 (a prime) elements apart never share a weight, so exchanging two unequal elements in one
row or one column of any published matrix (all have fewer than 65,521 rows and columns) always
fails, and a flipped, transposed or row-permuted layout passes only if the data solves one linear
equation. Each product stays below 2^31, so the int64 sum is exact up to 2^32 elements (8 GiB of
BF16, more than a whole publication). The position-free sum of format `/1` passed every permutation:
a loader that reversed every matrix's columns, or swapped `in_proj_b` and `in_proj_a` within
`in_proj_ba`, or one layer's tensor with another's, had every checksum equal; those are now
refused (`tests/test_flash_policy_weights.py`).

Measured when publications were files (MW smoke on kander, `flash-bench/tools/MW_REPORT.md`, the
live run's checkpoint 50, one server, the dithered merge): the trainer's sync applied the 5.44 GiB
publication from /data in 9.3 s
(the worker's read and hash 6.7 s, `load_weights` of 372 tensors 2.3 s, device checksums 0.04 s
for the position-free sums of then (the weighted checksum's GPU time is unmeasured), cache reset
0.002 s), the warm-up took 0.3 s and the window-start check 2 ms. A file with one extra
tensor, one with a tensor missing, and one with a flipped byte were each refused with HTTP 503 after
8-9 s (the whole file is hashed first) and left the engine poisoned. On 3,428 tokens of 16 seeded
T=1 completions of real ARC-2 prompts (5-30k tokens) drawn by the merged server, its logprobs sat
0.035-0.038 nats per token (mean absolute) from an engine serving the same adapter as LoRA and 0.036
from the learner, as close as that LoRA engine is to the learner (0.036) and to the merged
engine's own decode-versus-prefill distance (0.033); the learner scoring the published weights sat
0.035 from the server. Against the base model on the same tokens (0.15-0.16) the merged server
carried the adapter's per-token effect with slope 0.99 (correlation 0.96) relative to the LoRA
engine. Against that LoRA engine the merged engine prefilled 1.34x faster at 16 and at 32
concurrent requests and decoded 1.39x and 1.33x faster (23.7k and 1,232 tokens/s at 16; 23.4k and
2,042 at 32).

## Parity gate

### What it checks

Before each update the learner rescores real sampled tokens against the logprobs the server returned
(`_check_window_logprob_parity` in `training/ppo.py`), with the sampler's adapter: it keeps CPU
snapshots of the current and previous adapter (`_remember_parity_policy`) and, for a lag-one window,
loads the sampler's, scores and restores the current one, leaving optimizer state and RNG untouched.
All ranks score the same exact prefixes, each with its own forward, and each keeps its own scores
(they are the ones its training forward reproduces). The ranks' error statistics are pooled
(`_pool_parity_evidence`): the gate reads the worst rank's mean, p95, maximum and outlier count, and
a non-finite score on any rank fails it, so every rank expands, passes or fails together and a rank
whose scores are off fails the gate even when rank zero's pass. A failure stops the run.

`_parity_check_pool` rotates the checked request kind with the sampler policy,
`_PARITY_CONTEXT_CYCLE[(policy + 1) % 4]`: initial (step 0), reset (the first request after a
compaction), summary (a draw under the summary logit bias, reported as `biased_*`) and continuation,
falling back to the cheapest available evidence when a kind is missing. The gate scores 128 targets;
if any statistic misses its tolerance it extends the same prefixes to at least 1,000 targets and
decides on that, never on the first 128 alone. Under the overlapped schedule the zero adapter is
checked twice on initial samples, so the first trained adapter is always the first to meet a reset
sample: a jump there is the change of kind, not of adapter.

### Parity thresholds

The learner sits at the server's own decode-versus-prefill floor for every kind and adapter tested:
about 0.05 on initial prompts, 0.10-0.11 on continuations and long summary compactions, and
0.12-0.15 on short prune-mode compactions, which cut the context back to the current block (system
prompt, journal snapshot, state prompt) so the checked tokens open a fresh 3-4.5k-token context; the
floor follows the surprisal of the scored text. About 70% of the floor is MoE routing flips:
top-10-of-307 routing differs in at least one layer on nearly every token between any two
implementations, the server's own decode and prefill included, and replaying the server's routing in
the learner cuts the gap from 0.07-0.12 to 0.03-0.04. The rest is kernel noise on both sides.

Flash defaults (`ParityThresholds.flash()`, per-key overrides in
`training.flash.parity_thresholds`): mean |delta logprob| 0.25, p95 1.00, at most 1% of tokens
beyond 2.0 nats, `marginal_band_factor` 1.5; LoRA and full-weight modes keep 0.05 / 0.25 / 0.1%. At
the earlier 0.17 / 0.75 a correct learner failed 9-20% of reset checks (the server against itself
8-16%), and with about every other iteration checking a reset sample a run could expect a chance
stop within about ten iterations; at 0.25 / 1.00 with expansion before failing it fails 0-0.6%.
Initial, long-compaction and continuation checks failed at most 0.4% even at the old values.
Outliers stay at 1%: a T=1 draw of a very unlikely token can differ by several nats in BF16 without
any fault. Still open: kind-specific tolerances, or logging each kind's expected floor beside every
check, would make the gate sharper.

### Sharp edges of merged publication

The servers apply merged weights ([Serving](#merged-policy-weights)); where a fault could still hide
and how to look, from MW_REPORT.md (MW), L_HANDOFF.md (L), S2_REPORT.md (S2), R1_REPORT.md (R1) and
T_REPORT.md (T) in `kander:/data/personal/$USER/flash-bench/tools/`:

- **What is checked on every publication.** The label naming the manifest's sha256 and iteration;
  exactly the checkpoint's tensors for the declared targets with its shapes and dtype, before any
  byte moves; each tensor's home
  in the served model derived from the checkpoint's shapes and vLLM's packing order, not from the
  loader, with the packed parts tiling their matrix; every tensor landing once through the startup
  loader, inside its home's rows; every tensor's position-sensitive checksum recomputed from its
  home's rows in HF order after all copies, equal to the learner's; unchanged parameter storage; the
  publication's identity on every draw and at every window start. A dropped, stale, partial,
  mis-scaled, mis-placed or permuted publication, or bytes the broadcast changed, fails the
  publication, which no logprob comparison could promise at this floor.
- **What is not.** How the fork's kernels read a served matrix: the check establishes that each
  home holds the publication's matrix in the checkpoint's (HF) layout at vLLM's packed offsets,
  the same layout the startup loader gives the checkpoint itself; a kernel that expected another
  layout would misread the checkpoint too, which MW's comparison of a merged server with a
  LoRA-serving engine and the learner would show. A fork that packs differently (another part
  order, an interleaved layout) is refused, not served: rerun MW after a fork bump. And the
  checksum is one linear hash: a fault whose weighted changes cancel exactly passes it.
- **The merge is not either forward.** The learner's LoRA path adds `x @ A_bf16^T @ B_bf16^T *
  alpha/r` to the BF16 base product, vLLM's LoRA path folds the scale into B in BF16; the merged
  weight rounds `W + alpha/r B A` once, with a dither so that updates smaller than the rounding
  survive in expectation ([LoRA](#lora)). A nearest-rounded merge lost about a third of the
  adapter's effect at iteration 50 without failing any check: the parity gate's tolerance is far
  above that gap. MW measured the dithered merge as close to both LoRA paths as they are to each
  other.
- **What the gate sees.** A trained adapter moves sampled-token logprobs about 0.007 nats per token
  early in a run against a 0.05-0.15 floor (L), so the parity gate cannot see policy-sized serving
  faults; it catches what lies well above the floor: kernel faults (the GDN bug read 0.25), token,
  template or image misalignment, a wrong checkpoint. The checksums above cover the rest.
- **Nodes the group cannot span.** A cross-node publication needs InfiniBand between the learner's
  node and every rollout node. On 2026-10-10 dgx06 had only its four 100 Gb/s HCAs up (its 400 Gb/s
  compute fabric down): NCCL between dgx06 and dgx01 failed (`IBV_WC_RETRY_EXC_ERR`, probe 2926514)
  and a heterogeneous segment with its learner there hung in the group's warm-up all-reduce until it
  was cancelled (2926423; the initialization watchdog would have stopped rank 0 with exit 71 after
  30 minutes). Exclude such a node (`--exclude`), as dgx02 is excluded; the shared layout stays on
  one node and never uses the network.
- **The engine's own floor.** One e307 server's repeated base prefill of the same text differs by
  0.09-0.10 nats per token on average (p95 0.31, single tokens up to 1.8, never bitwise), and
  learner and server differ by 0.11-0.17 (S2, R1). No fixed-text check can be tighter.
- **Costs that scale with the model.** Each publication is 5.44 GiB (5.9 GB with routers), held on
  the learner's rank-0 GPU from the end of training to the rollout boundary and broadcast once to
  every server (NCCL delivers it to all ranks in one pass: the learner sends it once whatever the
  number of servers). Nothing goes to disk: earlier, servers on other nodes read each publication
  from /data (82-150 s per server cold), and then from a copy the learner pushed over srun (about
  20 s per node); now the whole publication, pause to resume with each tensor loaded as it arrives,
  takes about 1.1 s within a node and across nodes over InfiniBand (the broadcast about 1.0 s of it;
  toy runs 2926041 and 2926042, [Reference measurements](#reference-measurements)), where the raw
  broadcast of the same bytes takes 0.07-0.29 s (probes 2925913, 2925916): the per-tensor loads, not
  the network, set the pace. Of the policy, the run directory receives per iteration only the checkpoint
  (the PEFT adapter with its manifest, and the trainer state). The broadcast needs the learner at
  the boundary: in the overlapped schedule it waits for the window in flight before publishing, as
  the full-weight mode does.
- **When the fork pin, the checkpoint or the target set changes,** rerun MW once: publish from a
  real adapter to a server, score the same completions on that server, on an engine
  serving the adapter as LoRA, and on the learner; each pair should sit at the floor
  (`flash-bench/tools/mw_*`: `mw_learner.sbatch`, `mw_server.sbatch`, `mw_analyze.py`). The fork
  bump to `3bda1ba09` reran its server half (HP_REPORT.md, `flash-bench/tools/hp_*`): the same
  publication applied through the trainer's sync (9.8 s, all 372 checksums equal) and the extra-tensor
  control refused and poisoned the engine; on MW's 3,428 completion tokens the new server's prefill sat
  0.036 nats per token from the learner's LoRA forward and 0.034 from the learner on the published
  weights (MW's server: 0.036 / 0.035), 0.033 from the old fork's own prefill, and its decode 0.033 from
  its own prefill (MW: 0.033). The new fork serves Qwen4Exp without `torch.compile`, so these are
  the numbers to compare against after the next bump. The `mw_*`, `hp_*` and `pc_*` tools apply a
  publication file through `/policy_weights/apply`, which no longer exists: rerunning them means
  publishing over `/weights/*` from a learner rank (`StatelessNcclTrainerEndpoint` with a
  `VllmNcclWeightTransfer`, as `training/flash_backend.py` does), or a toy run.

## Training signal details that matter

**Episode packing and compaction.** With `training.episode_packing = true` the steps of an episode
nest token-for-token, so one packed sequence per segment holds every step's completion at its
original position, each span with its own advantage and logprobs. A step that does not extend its
predecessor raises, except at a flagged compaction (`segment_start`), which starts a new segment: N
compactions give N+1 sequences. The compacted prefix is prompt and therefore masked, so each segment
trains only its own post-compaction generations. A summary draw is trained once, in its own segment,
with the admitted logit bias (-100 on `agent_summary_forbidden_token_ids`) replayed at its targets
in every scoring path (policy, trust region, reference, parity); its text returns in the next
segment as masked prompt. At the first recipe's 32k window segments reached about 38k tokens; at
the deployment window they reach about the window plus a reply, 98k, one per micro-batch.

**Loss, budget and cycles.** GRPO advantages are the terminal reward minus the puzzle group's mean
over 8 attempts, over the group's population standard deviation (+1e-8). `token_mean` weighs every
generated token in the window equally, with one denominator across micro-batches and ranks. The
`auto` sample budget starts at full coverage (`initial_fraction` 1.0) and lowers the trained
fraction only if a trainer iteration exceeds 95% of a rollout iteration;
`extra_train_cycles_until_window` then reruns the window while a cycle still fits (stopping above
`extra_cycle_max_clip_fraction`, 0.10): in the recipe's run 6-10 extra cycles follow each pass. A
window whose primary clip fraction exceeds `max_window_clip_fraction` (0.5) is rejected.

**Rollout augmentation (opt-in).** `inference.augmentation.enabled` augments training rollouts,
never evaluation: on ARC-2 a dihedral transform and a permutation of the puzzle's own colors on every
grid, scored against the augmented test outputs; on ARC-3 a permutation of all 16 colors on every
frame, actions untouched. There is one draw per puzzle per window, a hash of `training.seed`, the
schedule index and the puzzle (`harness_support/augmentation.py`), so a GRPO group compares attempts
at one augmented puzzle and the next window redraws. It is a rollout setting only: the learner trains
on the tokens and images the workers produced, and parity is unchanged. It joins the checkpoint
identity, so a run cannot turn it on or off across a resume; the shipped recipe turns on both parts.
docs/arc2_training.md and docs/arc3_multimodal_training.md describe each harness's side.

**TensorBoard tags.** Parity results are under `train/logprob_parity/`; budget and cycles in
`train/sample_budget_fraction`, `train/train_to_collection_ratio` and `train/extra_cycles`.

| Tag (per window) | Value |
|---|---|
| `rollout/tokens_per_s`, `rollout/prompt_tokens_per_s`, `eval/tokens_per_s` | completion (or prompt) tokens over collection wall time; prompt tokens include cached context |
| `rollout/completion_tokens_per_rollout` | completion tokens per raw rollout: the generated tokens of all turns of one episode; `avg_tokens_per_rollout` in the iteration log and JSON |
| `train/tokens_per_s` | packed input tokens processed (all micro-batches, ranks, epochs and extra cycles; FSDP dummy slots and rejected cycles in, padding out) over `train/z_timing/optimizer_cycle_seconds` |
| `train/z_timing/optimizer_cycle_seconds` | primary plus extra cycle wall time, without sample building, parity, probe-entropy and trust-region checks or export (`optimization_seconds` keeps its wider interval) |
| `train/forward_time_s`, `train/backward_time_s`, `train/forward_backward_time_s` | CUDA-event intervals, largest rank total per cycle; backward includes recomputation and collectives |
| `rollout/compactions_per_rollout`, `eval/compactions_per_rollout` | mean count of `segment_start` steps after a rollout's first step, void attempts excluded |
| `train/expert_load_cv`, `train/expert_load_max_share`, `train/expert_load_entropy`, `train/experts_unused_fraction`; the same names under `rollout/` and `eval/` | expert load balance of the learner's routing, mean over the 48 layers: per-expert top-10 assignment counts' coefficient of variation (0 = balanced), busiest expert's count over the mean (1 = balanced), load entropy over log(307) (1 = uniform), fraction of experts no token reached. `…_worst_layer` gives the worst layer of the first three |

The compaction count is exact: the harness flags `segment_start` when `history_epoch` changes, the
ARC-3 duck never rewrites history at a level boundary, and the gateway lineage ledger
(`training/gateway_lineage.py`) admits a reset only with compaction evidence. Token counts come from
captured ids (usage metadata where eval captures none), collection time includes environment waits,
per-phase maxima can come from different ranks, and training and generation count different tokens.

**Expert load balance** (`training/expert_load.py`). Before routers train, we watch whether expert
usage drifts out of balance, with no balancing loss and nothing feeding the update. Each router's
`route_recorder` hook adds its top-10 choices to per-layer, per-expert counts on the GPU, by token
role: `train/` every real token of each real training forward of the window (every epoch and extra
cycle counts its tokens again), `rollout/` the positions whose next token the policy sampled (the
tokens the loss scores), `eval/` the same positions in a no-grad pass over eval episodes. A count is
armed only around a real micro-batch's policy forward, so checkpoint recomputation, FSDP padding
slots, and parity, trust-region and probe-entropy forwards never count. Each window pools the counts
with one `all_reduce`. These are the learner's routing of those tokens; the servers' routing differs
only by numerical flips at the parity floor. `rollout/` tags sit on the rollout axis like every
`rollout/` scalar, `train/` and `eval/` on the trainer iteration, and one `expert load` log line per
window adds the token counts and the worst layer's index. `training.flash.load_balance_metrics =
false` removes the hooks. Official eval runs in its own process, so after each cycle it hands up to
`training.flash.load_balance_eval_episodes` (default 8; 0 disables) episodes, packed as training
packs them, to `<run>/expert_load_eval/`; at its next window end the learner scores the newest
cycle with its current policy (sharded across ranks, `padded_rank_shard`) and deletes the files.
The shipped recipe has official eval off, so `eval/` expert load tags do not appear.

Measured on the real learner (4 B200s, the live run's 8 probe samples): counting costs nothing
visible (median cycle 13.600 s off, 13.608 s on); the eval pass runs at 38k tokens/s, about 10 s
per window for 8 episodes of ~50k packed tokens. e307 starts far from balanced on ARC-2 tokens
(`uniform` in its name is the pruning rule, the same expert count per layer): `train/` cv 1.30,
max share 10.0, entropy 0.90, 0.26% unused; `rollout/` 1.47, 11.8, 0.88; worst layer 15, where one
expert takes 62% of all tokens, mostly grid characters (our unpruned ARC-3 replay has the same
shape). Read drift against these starting values, not against 1.

## Operations

- **Stop cleanly**: `touch <run dir>/STOP_AUTO_RELAUNCH`, then `scancel --batch
  --signal=TERM <jobid>`; remove the marker before resubmitting, or the next segment ends at once. The script signals the learner ranks directly (TERM to torchrun would
  start its short kill timeout) and gives them 540 s to finish the iteration and checkpoint before
  stopping the servers. Slurm reports FAILED, exit 143; the run stays resumable.
- **Artifacts** (run directory on /data): `trainer.log`, `logs.log`, `server_<i>.log`, launch,
  frozen and effective configs, `source.tar`, `checkpoints/`, `tensorboard/`.
  Both Flash recipes set `training.checkpoint_keep_last: null`, keeping every
  `checkpoints/adapter_iter_NNNNNN/` (LoRA, routers and `policy_weights.json`) and
  `checkpoints/trainer_state_iter_NNNNNN.pt` without pruning or hard/pinned caps.
  Budget about 0.9 GB per iteration (200 MB adapter + 690 MB trainer state),
  roughly 32 GB/day at 40-minute iterations. To bound retention, set an integer >= 2;
  the prior trainer state must survive publishing/evaluation before `training_state.json`
  advances. The default `6` retains the newest six trainer states plus the initial
  `trainer_state_iter_-00001.pt`, and the newest six prunable adapters with the two-hour
  minimum age, hard cap of 40 prunable adapters and exemption for the newest 40 `PINNED`
  snapshots. Retention may change on resume; it is outside checkpoint/recipe identity.
- **TensorBoard**: one folder per arm (by default the submit helper's
  `/data/personal/$USER/arc-training-runs/<config stem>-<date>/`), with a `README.md` describing the
  arm and the run directory `lora/`; register `<arm>:<arm folder>/lora/tensorboard` in the
  TensorBoard server's `--logdir_spec` (`make tensorboard TENSORBOARD_LOGDIR_SPEC=...`).
- **Follow-ups**: the harness pins are harness `main` of 2026-10-09 (arc2-platypus `f380ffd`,
  arc3-platypus `e53d822d`, fork `3bda1ba09`); a later bump needs new harness and ARC-2 worker
  environments (and a relock of `environments/arc2-worker-vllm-fork` against the harness lock) and
  the merged-publication re-check above.
  With `cleanup_rollout_artifacts = false` a segment ends by copying retained rollouts file by file
  to /data, which takes minutes. Servers idled about 19 s per window boundary, 15 s of it while the
  learner read the server log to confirm each adapter load; the boundary now takes the NCCL
  publication (see [Reference measurements](#reference-measurements)) and an HTTP confirmation. Rank imbalance
  costs an estimated 20-25% of forward/backward time, and the trust-region and probe-entropy
  forwards repeat the same sequences on every rank (about 85 s per iteration).

## Reference measurements

**Parity floor by request kind**, mean |delta logprob| over sampled tokens. *Provenance: replicas of
a production server against one learner rank, 55 prompts from a text-only ARC-3 run, r32 adapters.*

| Kind | Learner vs sampler (4 adapters) | Server prefill vs own decode |
|---|---:|---:|
| initial (step 0) | 0.043-0.062 | 0.049-0.054 |
| reset, short prune-mode compaction | 0.118-0.147 | 0.129-0.137 |
| reset, long summary compaction | 0.108-0.115 | 0.106-0.111 |
| continuation, 8-16k tokens | 0.102-0.109 | 0.100-0.107 |

**Gate false-failure rate on short prune-mode compactions.** *Provenance: the gate emulated on 69
such samples per adapter (rates under 0.5% unresolved); always-expand from 2,000 simulated checks.*

| Mean / p95 / outliers | Learner, 128-target rule | Server vs itself | Learner, always expand |
|---|---:|---:|---:|
| 0.17 / 0.75 / 1% (earlier default) | 8.9-19.5% | 7.9-15.7% | |
| 0.20 / 0.90 / 1% | 1.6-4.8% | 0.2-3.4% | 0.2-1.8% |
| 0.25 / 1.00 / 1% (current default) | 0-2.0% | 0-1.9% | 0-0.6% |
| 0.30 / 1.20 / 2% | 0-1.9% | 0 | 0 |

**Throughput and memory.** *Provenance: the recipe's run at 64 attempts per window (steady-state
iterations 5-25; learner on 4 B200s, 16 attempts per server, BF16 KV); a benchmark of
`FlashPPOTrainer` as in the recipe on 15 packed ARC-2 samples of 10-35k tokens; a serving benchmark
on one B200 with r16 adapters; the MW smoke (live run's checkpoint 50, one B200 server). The run and
the learner benchmark kept the retired NVFP4 expert storage; BF16 experts run the same optimizer
cycle 1.60x faster, and the memory after build is theirs (job 2923707,
[Expert storage](#expert-storage)).*

| Quantity | Value |
|---|---:|
| learner: packed tokens per pass (126 sequences) | 3.23M (1.96M completion) |
| learner: primary forward/backward pass | 292 s, 11,180 tokens/s (2,800 per GPU) |
| learner: one pass, samples to adapter export | 387 s of a 978 s iteration |
| learner: slowdown on the reference path (real 28-micro-batch cycle; one 34.5k-token micro-batch) / with Flex QSA only | 2.36x; 2.51x / 1.57x |
| learner: per-rank memory after build; peak at 32.5-37.9k tokens | 38.2 GiB; 98-103 GiB allocated, about 145 reserved, of 178 (NVFP4) |
| server: weights on GPU / PLE table on host | 57.86 GiB / 47.68 GB |
| server: KV pool; peak use in the run | 108.68 GiB, 3.98M tokens; 10.3-11.1% |
| server at fork `3bda1ba09`, no LoRA (HP smoke): KV pool; prefill / decode tokens/s at 16 and 32 concurrent | 117.84 GiB, 4.32M tokens; 23.7k, 23.6k / 1,241, 2,104 (MW at `077920f3`: 23.7k, 23.4k / 1,232, 2,042) |
| server: `--enable-lora` without an active adapter | 0-3% slower; KV 4.32M to 3.99M tokens |
| server: active r16 adapter vs base model | decode 20-26%, prefill 23-30% slower |
| learner: merged publication (372 tensors, 5.44 GiB); gather+merge (when it was then written to /data: 11.3 s more) | 3.2 s |
| NCCL broadcast of 5.62 GiB in 360 tensors, learner rank 0 to vLLM-fork receivers (probes 2925913, 2925916) | 0.07-0.23 s to two receivers in one node; 0.23-0.29 s to one on another node (InfiniBand) |
| publication in the toy runs (372 tensors, 5.44 GiB; one node, 4 servers, jobs 2926041 and 2926545; learner and 2 servers on two nodes, 2926042, 2926512 and 2926716): gather+merge on 4 / 2 ranks; pause to resume (broadcast inside); each server's verification; confirmation (info and warm-up) of all servers | 0.25-0.41 / 0.24-0.40 s; 1.13 / 1.13-1.14 s (0.95 / 0.99-1.00 s); 0.06-0.12 s; 0.09 / 0.05 s. The first publication of a segment adds the group's initialization: 3.8-4.1 / 2.2-2.3 s in all. A resumed segment republished each restored adapter with the identity its run recorded |
| parity on windows sampled by NCCL-published trained policies (toy runs, 128 tokens) | one node, policy 1, lag 0: mean 0.063, p95 0.31; two nodes, policy 0, lag 1: mean 0.154, p95 0.65; both passed |
| server: apply from /data, the file route this replaced (read+hash / load / checksums); warm-up | 9.3 s (6.7 / 2.3 / 0.04 s, format `/1` sums); 0.3 s |
| server: merged weights vs r16 LoRA engine, prefill / decode tokens/s at 16 and 32 concurrent | 1.34x, 1.34x / 1.39x, 1.33x |
| server: generation over a window; prefix-cache hit rate | 532-547 tokens/s; 0.88 |
| server: decode tokens/s at 1 / 8 / 16 running (run); at 16 / 32 / 64 concurrent (benchmark) | 69 / 435 / 727; 890 / 1,640 / 2,597 |
