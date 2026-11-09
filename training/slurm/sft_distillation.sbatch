#!/usr/bin/env bash
#SBATCH --job-name=arc-router-distill
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=b200:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
REPO=${1:?repository worktree}
REVISION=${2:?expected git commit}
CONFIG=${3:?immutable attempt config}
RESULT=${4:?new result directory}
CONFIG_SHA=${5:?expected configuration hash}
[[ "$(sha256sum "$CONFIG" | cut -d " " -f 1)" == "$CONFIG_SHA" ]]
cd "$REPO"
git cat-file -e "$REVISION^{commit}"
. "${ARC_JOB_ENV:-/etc/kander/job-env.sh}"
TASK="$SCRATCH_DIR/arc-router-distillation-$SLURM_JOB_ID"
mkdir -p "$TASK/source"
# Run the committed source snapshot, immune to subsequent worktree edits.
git archive "$REVISION" | tar -x -C "$TASK/source"
cd "$TASK/source"
unset VIRTUAL_ENV PYTHONPATH UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export UV_CACHE_DIR="/cache/users/$(id -u)/uv"
uv venv "$TASK/venv" --python 3.12
PY="$TASK/venv/bin/python"
uv export --frozen --no-dev --extra flash-distillation --no-emit-workspace \
  --format requirements-txt --output-file "$TASK/requirements.txt" >/dev/null
uv pip install --python "$PY" -r "$TASK/requirements.txt"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export TRITON_CACHE_DIR="$TASK/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/inductor" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ARC_DISTILLATION_GIT_COMMIT="$REVISION"
# Variable-length whole-sequence batches otherwise strand large cached blocks.
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
"$PY" -m unittest discover -s tests/distillation
# Kernel cache of a kernel_mode "fast" run. The fast stack autotunes FLA's
# convolutions per 1024-token length bucket and compiles its fused kernels on
# first use: with fresh caches its first updates took 3-6x as long as warm ones
# (reference 2x; training/FLASH_RL.md, "Kernel mode").
# The run's segments share them through <run dir>/kernel_cache.tar.zst, unpacked
# at one per-run path (Triton's cache index holds absolute paths). The unit
# tests above keep the job's own cache: autotune picks must not cross workloads.
# A "reference" run keeps the per-job cache, as its results were produced.
restore_kernel_cache() {
  rm -rf -- "$KERNEL_CACHE"
  mkdir -p "$KERNEL_CACHE/triton" "$KERNEL_CACHE/inductor"
  [[ -f "$KERNEL_ARCHIVE" ]] || return 0
  if timeout -k 30s 10m tar -I zstd -xf "$KERNEL_ARCHIVE" -C "$KERNEL_CACHE"; then
    echo "Restored the run's kernel cache from $KERNEL_ARCHIVE"
  else
    echo "Could not restore $KERNEL_ARCHIVE; training starts with an empty kernel cache" >&2
    rm -rf -- "$KERNEL_CACHE"
    mkdir -p "$KERNEL_CACHE/triton" "$KERNEL_CACHE/inductor"
  fi
}
save_kernel_cache() {
  if timeout -k 30s 10m tar -I zstd -cf "$KERNEL_ARCHIVE.tmp" -C "$KERNEL_CACHE" triton inductor &&
    mv -f -- "$KERNEL_ARCHIVE.tmp" "$KERNEL_ARCHIVE"; then
    echo "Saved the run's kernel cache to $KERNEL_ARCHIVE"
  else
    rm -f -- "$KERNEL_ARCHIVE.tmp"
    echo "Could not save the kernel cache; $KERNEL_ARCHIVE is unchanged" >&2
  fi
  case "$KERNEL_CACHE" in "/cache/users/$(id -u)/arc-distillation/compile/run-"*) rm -rf -- "$KERNEL_CACHE" ;; esac
}
KERNEL_MODE=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("kernel_mode", "reference"))' "$CONFIG")
if [[ "$KERNEL_MODE" == fast ]]; then
  # The launcher's result directory is <run dir>/attemptNNN/training.
  KERNEL_ARCHIVE="$(dirname "$(dirname "$RESULT")")/kernel_cache.tar.zst"
  KERNEL_CACHE="/cache/users/$(id -u)/arc-distillation/compile/run-$(printf '%s' "$KERNEL_ARCHIVE" | sha256sum | cut -c1-16)"
  restore_kernel_cache
  export TRITON_CACHE_DIR="$KERNEL_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$KERNEL_CACHE/inductor"
  trap save_kernel_cache EXIT
fi
"$PY" -m sft.distillation.train --config "$CONFIG" --result "$RESULT" --stage-dir "$TASK/model-shards" --gpu-telemetry
