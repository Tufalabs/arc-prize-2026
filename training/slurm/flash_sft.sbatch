#!/bin/bash
#SBATCH --job-name=flash-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=239G
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
# Flash SFT (training/flash_sft.py) on one node: stage the learner environment, then torchrun one rank per GPU.
# The caller sets account, partition, GPUs, CPUs, time, --chdir and --output on the sbatch line.
# Host RAM scales with requested GPUs: 239 GiB each (956 GiB for four GPUs).
# Override with sbatch --mem-per-gpu=<budget> for another node/qualified recipe.
# Required environment:
#   FLASH_SFT_SOURCE=/path/to/source   a snapshot (git archive) of the revision to run, not a dev checkout
#   FLASH_SFT_CONFIG=CONFIG.json       the run's configuration (configs/flash_sft/*.json, every key explicit)
#   FLASH_SFT_RUN_DIR=/data/.../<run>  checkpoints, metrics.jsonl and config.json; reuse it to resume
#   ARC3_FLASH_VENV_TARBALL=/data/personal/$USER/arc-training-envs/trainer-flash-distillation-<key>.tar.zst
#     built by scripts/build_env.sbatch trainer <revision> (docs/environments.md); the adjacent .json
#     manifest is required.
# Optional:
#   FLASH_SFT_STAGE_MODEL=1            copy config["model"]'s safetensors shards to node-local scratch first
#     (sft.distillation.staging: copied, sha256-verified) and read them from there; default 0 reads /data.
# A resubmission with the same run directory and configuration resumes from its latest checkpoint.
set -euo pipefail
unset UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
. /etc/kander/job-env.sh
umask 027
SOURCE="$(realpath "${FLASH_SFT_SOURCE:?set FLASH_SFT_SOURCE}")"
CONFIG="$(realpath "${FLASH_SFT_CONFIG:?set FLASH_SFT_CONFIG}")"
RUN_DIR="$(realpath -m "${FLASH_SFT_RUN_DIR:?set FLASH_SFT_RUN_DIR}")"
[[ "$RUN_DIR" == /data/* ]] || { echo "FLASH_SFT_RUN_DIR must be under /data" >&2; exit 1; }
[[ -f "$SOURCE/training/flash_sft.py" ]] || { echo "$SOURCE holds no training/flash_sft.py" >&2; exit 1; }
TARBALL="${ARC3_FLASH_VENV_TARBALL:?set ARC3_FLASH_VENV_TARBALL}"
KEY="$(basename "$TARBALL" .tar.zst)"; KEY="${KEY##*-}"
. "$SOURCE/scripts/stage_venv.sh"
stage_venv "$TARBALL" "$KEY" FLASH_PYTHON
mkdir -p "$RUN_DIR" "$SCRATCH_DIR/triton" "$SCRATCH_DIR/inductor"
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH_DIR/inductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$SOURCE" PYTHONHASHSEED=0
case "${FLASH_SFT_STAGE_MODEL:-0}" in
  0) unset FLASH_SFT_STAGED_MODEL ;;
  1)
    MODEL="$("$FLASH_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$CONFIG")"
    export FLASH_SFT_STAGED_MODEL="$SCRATCH_DIR/model-staged"
    echo "flash_sft: staging $MODEL -> $FLASH_SFT_STAGED_MODEL ($(date +%T))"
    "$FLASH_PYTHON" -m sft.distillation.staging --source "$MODEL" --destination "$FLASH_SFT_STAGED_MODEL"
    echo "flash_sft: staged ($(date +%T))"
    ;;
  *) echo "FLASH_SFT_STAGE_MODEL must be 0 or 1" >&2; exit 1 ;;
esac
GPUS="$(nvidia-smi -L | wc -l)"
echo "flash_sft: source $SOURCE, config $CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)), run $RUN_DIR, $GPUS GPUs, venv $KEY"
cd "$SOURCE"
"$FLASH_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$GPUS" \
  -m training.flash_sft --config "$CONFIG" --run-dir "$RUN_DIR" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"' TERM INT
wait "$child"
