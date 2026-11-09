#!/usr/bin/env bash
#SBATCH --job-name=sft-train
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=300G
#SBATCH --time=10:00:00
#SBATCH --output=/data/personal/%u/ARC3-Training/slurm/slurm-%x-%j.out
#
# Per-action SFT (Qwen3.5-9B LoRA) from TAAF traces, following the Kander
# storage pattern:
#   durable on /data  ->  stage in to $SCRATCH_DIR  ->  work with caches on /cache
#   ->  stage the checkpoint back to /data.
# Override the data with:  SFT_CONFIG=/path/to.json sbatch sft_train.sbatch

set -euo pipefail

# Drop any UV/HF cache paths leaked from ~/.bashrc via --export=ALL (e.g. the
# login-only UV_CACHE_DIR=/scratch/$USER/uv) so the managed defaults apply.
unset UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME VIRTUAL_ENV

# Kander-managed job env: sets SCRATCH_DIR (/scratch/jobs/$JOBID), UV_CACHE_DIR
# (/cache/users/$UID/uv), HF_HOME, TORCH_HOME, TMPDIR -> all node-local.
. /etc/kander/job-env.sh

# --export=ALL leaks the submitting shell's env: surface the effective
# empty-completion policy so a lingering =skip can't apply invisibly.
echo ">> SFT_ON_EMPTY_COMPLETION=${SFT_ON_EMPTY_COMPLETION:-error (repo default)}"

# Guarantee a valid node-local uv cache regardless of what job-env.sh exported.
export UV_CACHE_DIR="/cache/users/$(id -u)/uv"
mkdir -p "${UV_CACHE_DIR}"

# Worktree support: the submitting script passes SFT_PROJECT_DIR so this job
# uses the checkout it was launched from. This cannot self-locate the way the
# submit scripts do -- Slurm runs batch scripts from a spool copy, so
# BASH_SOURCE here is the spool path, not the repo.
PROJECT_DIR="${SFT_PROJECT_DIR:-/data/personal/${USER}/ARC3-Training}"
SFT_CONFIG="${SFT_CONFIG:-${PROJECT_DIR}/configs/sft.json}"
SFT_UV_EXTRA_ARGS=()
if [ -n "${SFT_EXTRA_ARGS:-}" ]; then
  read -r -a SFT_UV_EXTRA_ARGS <<< "${SFT_EXTRA_ARGS}"
fi
RUN_DIR="${SCRATCH_DIR}/ARC3-Training"

# Reuse the already-downloaded Qwen3.5-9B snapshot on /data (large sequential
# reads from NFS are fine; avoids a 19 GB re-download to node-local cache).
export HF_HOME="/data/personal/${USER}/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# On a dataset-cache miss (config `dataset_cache_dir`), sft/dataset_cache.py
# stages the raw trace dirs here (node-local) with a parallel copier and
# builds from the local copies, then packs the result back to the durable
# cache on /data. On a cache hit the raw traces are never touched.
export PYTHONUNBUFFERED=1
export SFT_STAGE_DIR="${SCRATCH_DIR}/trace_stage"

echo ">> job ${SLURM_JOB_ID} on $(hostname)"
echo ">> project=${PROJECT_DIR}  config=${SFT_CONFIG}"

# Honor the config run_name for the Slurm job name. submit_sft.sh already sets
# it at submission (so the log filename picks it up via %x); this also covers a
# bare `sbatch sft_train.sbatch` where only the header default would apply.
RUN_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name") or "")' "${SFT_CONFIG}" 2>/dev/null || true)"
if [ -n "${RUN_NAME}" ]; then
  scontrol update jobid="${SLURM_JOB_ID}" name="${RUN_NAME}" >/dev/null 2>&1 || true
  echo ">> run_name=${RUN_NAME}"
fi
# One durable run dir per job, named by the config run_name so runs are
# recognizable in runs/ and TensorBoard and sort naturally by job id: runs/<jobid>-<run_name>/.
RUN_TAG="${SLURM_JOB_ID}-${RUN_NAME:-sft}"
# Checkpoints go STRAIGHT to durable /data: with save-on-every-eval the whole
# point is surviving a wall-kill, and the end-of-job rsync stage-out never runs
# on a killed job. Checkpoint files are few large sequential writes (NFS-OK).
OUTPUT_DIR="${PROJECT_DIR}/runs/${RUN_TAG}/outputs"
# TensorBoard events also on /data (tiny scalar writes), readable live from the
# login node. transformers 5.x ignores the SFTConfig `logging_dir` arg for TB
# and reads this env var instead (the arg is deprecated), so set it explicitly.
TB_DIR="${PROJECT_DIR}/runs/${RUN_TAG}/tensorboard"
export TENSORBOARD_LOGGING_DIR="${TB_DIR}"

echo ">> scratch=${RUN_DIR}  out=${OUTPUT_DIR}  uv_cache=${UV_CACHE_DIR:-unset}"

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${TB_DIR}"

# ---- stage in: repo -> node-local scratch (code only) ----
rsync -a "${PROJECT_DIR}/" "${RUN_DIR}/" \
  --exclude .git --exclude .venv --exclude .venvs --exclude node_modules \
  --exclude __pycache__ --exclude runs --exclude slurm \
  --exclude dataset_cache

# Copy the config into scratch and repoint its output_dir to node-local storage
# so the trainer writes checkpoints fast, not onto NFS.
cp "${SFT_CONFIG}" "${RUN_DIR}/_sft_run.json"
python3 - "${RUN_DIR}/_sft_run.json" "${OUTPUT_DIR}" "${TB_DIR}" <<'PY'
import json, sys
path, out, tb = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(path))
cfg["output_dir"] = out
cfg["logging_dir"] = tb
json.dump(cfg, open(path, "w"), indent=2)
print("output_dir ->", out)
print("logging_dir ->", tb)
PY

cd "${RUN_DIR}"

# Build the venv on node-local scratch, OUTSIDE the repo tree (the repo ships a
# .venv symlink to a login-only path) and never on /data; cache on /cache.
export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/venv"

# Start from an empty global git config so no login-only setting (e.g. a
# url rewrite) leaks into the job. When SFT_EXTRA_ARGS enables sft-rearc, its
# private dependency authenticates through the SSH key in the NFS-mounted
# home directory.
export GIT_CONFIG_GLOBAL="${SCRATCH_DIR}/gitconfig"
: > "${GIT_CONFIG_GLOBAL}"
export GIT_TERMINAL_PROMPT=0

# ---- venv + fast linear-attention kernels ----
# Build the full training venv once with uv, install the fast kernels on top,
# then NEVER run uv against the venv again: uv run/sync enforce the lock
# exactly and would strip the extra packages.
env -u VIRTUAL_ENV uv sync --locked --extra sft --extra sft-train "${SFT_UV_EXTRA_ARGS[@]}"
PY="${UV_PROJECT_ENVIRONMENT}/bin/python"

# Qwen3.5's linear-attention layers need flash-linear-attention +
# causal-conv1d for the fused fast path (~3-4x faster training than the torch
# fallback). fla is pure Triton; causal-conv1d is a prebuilt B200 wheel
# (py312/torch2.11/cu130, built for the RL runs) on each node's local /cache,
# with a durable /data copy as fallback for nodes that lack it.
# ARC3_FAST_KERNELS=0 skips the install and runs the slow torch fallback.
ARC3_CAUSAL_CONV1D_WHEEL="${ARC3_CAUSAL_CONV1D_WHEEL:-/cache/shared/research/arc-rl/wheels/py312-torch2.11.0-cu130-b200/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl}"
if [ ! -f "${ARC3_CAUSAL_CONV1D_WHEEL}" ]; then
  ARC3_CAUSAL_CONV1D_WHEEL="/data/personal/${USER}/wheels/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl"
fi
case "${ARC3_FAST_KERNELS:-1}" in
  0|false|FALSE|no|NO|off|OFF)
    echo ">> ARC3_FAST_KERNELS disabled; using slow torch fallback"
    ;;
  *)
    if [ ! -f "${ARC3_CAUSAL_CONV1D_WHEEL}" ]; then
      echo ">> causal-conv1d wheel not found (node /cache or /data fallback); set ARC3_FAST_KERNELS=0 to run without fast kernels" >&2
      exit 1
    fi
    env -u VIRTUAL_ENV uv pip install --python "${PY}" \
      flash-linear-attention==0.5.1 fla-core==0.5.1 "${ARC3_CAUSAL_CONV1D_WHEEL}"
    "${PY}" -c 'import causal_conv1d, fla; print(">> fast kernels ready: causal_conv1d", causal_conv1d.__version__, "fla", fla.__version__)'
    ;;
esac
# Runtime knobs the RL pipeline sets for fla (training/cli.py
# _configure_fast_kernel_defaults); Triton JIT cache on node-local scratch.
export FLA_TILELANG=0 FLA_CONV_BACKEND=triton
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton"

# ---- data sanity (GPU-light, no model load) then train one epoch ----
# Direct venv python, NOT make/uv run, so the kernel install survives.
"${PY}" -m sft.train --config "${RUN_DIR}/_sft_run.json" --check-data
# Multi-GPU (submit with --gpus=b200:N, e.g. via SFT_GPUS in submit_sft_chain.sh):
# torchrun spawns one rank per GPU; HF Trainer picks up the env and runs DDP.
# The packs must already exist (prepack job or the --check-data pass above) --
# ranks only ever read the dataset cache.
NGPU="${SLURM_GPUS_ON_NODE:-1}"
if [ "${NGPU}" -gt 1 ]; then
  echo ">> launching DDP on ${NGPU} GPUs"
  "${UV_PROJECT_ENVIRONMENT}/bin/torchrun" --standalone --nproc-per-node="${NGPU}" \
    -m sft.train --config "${RUN_DIR}/_sft_run.json"
else
  "${PY}" -m sft.train --config "${RUN_DIR}/_sft_run.json"
fi

echo ">> checkpoints + final model in ${OUTPUT_DIR}"
