#!/usr/bin/env bash
#SBATCH --job-name=sft-prepack
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/data/personal/%u/ARC3-Training/slurm/slurm-%x-%j.out
#
# CPU-only dataset pre-pack: runs `sft.train --check-data` so both packs land
# in the durable dataset_cache before the (dependent) GPU training job starts.
# The streaming build shards traces across SFT_BUILD_NUM_PROC worker procs --
# concurrency is what beats NFS small-file latency on compute nodes (measured:
# ~50 files/s serial vs >1k files/s at 32+ readers on a quiet node).

set -euo pipefail

unset UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME VIRTUAL_ENV SFT_STAGE_DIR
. /etc/kander/job-env.sh

# --export=ALL leaks the submitting shell's env: surface the effective
# empty-completion policy so a lingering =skip can't apply invisibly.
echo ">> SFT_ON_EMPTY_COMPLETION=${SFT_ON_EMPTY_COMPLETION:-error (repo default)}"
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
export PYTHONUNBUFFERED=1
export SFT_BUILD_NUM_PROC="${SFT_BUILD_NUM_PROC:-32}"
# Arrow build workspace on node-local scratch (NFS + parallel shard writers
# race on cleanup); only the finished pack lands on /data.
export SFT_BUILD_WORK_DIR="${SCRATCH_DIR}/build"

echo ">> prepack job ${SLURM_JOB_ID} on $(hostname): config=${SFT_CONFIG} num_proc=${SFT_BUILD_NUM_PROC}"

cd "${PROJECT_DIR}"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/venv"
export GIT_CONFIG_GLOBAL="${SCRATCH_DIR}/gitconfig"
: > "${GIT_CONFIG_GLOBAL}"
export GIT_TERMINAL_PROMPT=0

env -u VIRTUAL_ENV uv sync --locked --extra sft "${SFT_UV_EXTRA_ARGS[@]}"
"${UV_PROJECT_ENVIRONMENT}/bin/python" -m sft.train --config "${SFT_CONFIG}" --check-data
echo ">> packs ready in dataset_cache"
