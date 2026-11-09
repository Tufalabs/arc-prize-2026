# `sft/` — per-action SFT from TAAF traces

The SFT data and training loop are separate from PPO, but they are part of the
same Python project and dependency lock. Runtime adapter export deliberately
reuses `training.runtime_lora` so SFT and RL checkpoints obey the same serving
contract. Inputs are TAAF traces produced by `tufa-arc-agi-framework`, plus
optional benchmark metadata and re_arc dataset selections used for filtering.

The agreed ARC3 SFT holdouts and their tagged derivatives live in
[`holdouts.py`](holdouts.py). This adds `crane_dipper-0001`, tagged
`official-reimplemented`, to the training exclusions because it reimplements
the held-out CD82 game. Both the TRL source-selection entrypoint and native
`training.flash_sft` reject training inputs containing those game families,
including unversioned IDs and other versions. Exclude them during preparation;
validation can still load them. The rollout protocol remains five games.
`sliders-0001` is a reimplementation of S5I5, which is not a holdout, so this
policy does not exclude it.

Provenance: re-arc-3's [CD82 metadata](../../re-arc-3/re_arc/environment_files/cd82/fb555c5d/metadata.json)
and [Crane Dipper metadata](../../re-arc-3/re_arc/environment_files/crane_dipper/0001/metadata.json).

## Layout

| File | Role |
|---|---|
| `taaf_adapter.py` | Load TAAF traces → TRL `Dataset` (retemplate / token_exact); root discovery |
| `dataset.py` | Shared helpers: image resolve, message projection, Dataset build |
| `dataset_cache.py` | Packed-dataset Arrow cache + parallel trace staging (pre-pack once, train many) |
| `selection.py` | Trace selection by game / tag / score / optional named re_arc dataset, with held-out eval |
| `config.py` | `SFTRunConfig` (JSON → dataclass): `trace_root`, `datasets`, `training_mode`, … |
| `lora_wrap.py` | LoRA wrapping + setup-time census (language-model projections only) |
| `train.py` | TRL `SFTTrainer` entrypoint; branches on resolved mode; `--check-data` |
| `export_runtime.py` | Export checkpoints to vLLM hot-loadable `checkpoint-N-runtime/` dirs |

## Modes (`training_mode`)

- `retemplate` — messages + single-turn completion; loss on completion only.
  Re-tokenized by the target template. Works for vision.
- `token_exact` — pre-tokenized rows from TAAF `input_token_ids`/`output_token_ids`
  (prompt masked). Text-only. Requires token ids. Input-token fidelity is
  verified at capture time by the trace writer (ARC3-Inference records
  `input_token_reconstruction_status` per step, tools included in the render);
  training trusts the stored ids.
- `auto` (default) — `token_exact` when every step has token ids and no images,
  else `retemplate`.

## Run

```bash
# inspect what the trainer will receive for a trace root — no model load
make sft-check SFT_CONFIG=configs/sft.json

# add named re_arc dataset resolution when DatasetSpec.dataset is used
make sft-check SFT_CONFIG=configs/sft.json SFT_EXTRA_ARGS=--extra=sft-rearc

# train (needs trl + a checkpoint; GPU recommended)
make sft-train SFT_CONFIG=configs/sft.json

# export checkpoints for vLLM hot-load serving
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$PWD/.venvs/sft" \
  uv run --locked --no-sync python -m sft.export_runtime --run-dir runs/<name> --all
```

Point `trace_root` in `configs/sft.json` at a TAAF trace root, or use the
`datasets` selection specs for filtered multi-run training. The `sft-rearc`
extra is required only when a selection uses `DatasetSpec.dataset` without an
explicit `re_arc_datasets_dir`; pass `SFT_EXTRA_ARGS=--extra=sft-rearc` for such
a config. These Make targets use `.venvs/sft`, isolated from the RL/server
runtime environment. On Kander, submit via
`slurm/submit_sft_chain.sh <config>` (CPU pre-pack job, then GPU training).
The Slurm jobs install from `uv.lock`; opt into the private dataset dependency
only when needed:

```bash
SFT_EXTRA_ARGS=--extra=sft-rearc slurm/submit_sft_chain.sh <config>
```

## Native Flash SFT host memory

The native launcher `slurm/flash_sft.sbatch` and the mixed CE+KL launcher
`scripts/flash_mixed_cosine120/train.sbatch` request **239 GiB of host RAM per
requested GPU** with `#SBATCH --mem-per-gpu=239G`. Both are single-node jobs.
This gives one-, two-, four- and eight-GPU jobs 239, 478, 956 and 1,912 GiB
respectively, so a job's default RAM reservation follows its GPU share.

Kander's eight-B200 nodes advertise 1,960,720 MiB (approximately 1,914 GiB) of
schedulable RAM. Eight GPU shares therefore reserve 1,912 GiB, leaving
2,832 MiB of scheduler headroom, whether used by one eight-GPU run or two
four-GPU runs. The mixed launcher previously requested 350 GiB per GPU,
reserving 1,400 GiB for four GPUs and preventing two such runs sharing a node.
Use `sbatch --mem-per-gpu=<budget>` to override the default for another node
type or a recipe with a separately qualified memory requirement.

This sets the scheduler reservation and memory limit; it does not reduce the
trainer's allocations. Host usage includes activation-offload buffers, optimizer
state and file cache. File-backed cache can be reclaimed, but active pinned
buffers cannot. The 956-GiB limit still needs a longest-row GPU qualification
for the mixed CE+KL and original-BF16 expert recipes before production use.
Changing these launchers affects new submissions; existing Slurm allocations
and frozen launch artifacts retain their current budgets.
