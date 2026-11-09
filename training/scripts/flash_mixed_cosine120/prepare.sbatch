#!/bin/bash
#SBATCH --job-name=mixed120-data
#SBATCH --account=arc
#SBATCH --partition=arc
#SBATCH --qos=arc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/preparation/job-%j.log
set -euo pipefail
TASK_ROOT=${MIXED_TASK_ROOT:-/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/artifacts}
TASK_SCRATCH=/scratch/jobs/$SLURM_JOB_ID/mixed120
mkdir -p "$TASK_SCRATCH/source" "$TASK_SCRATCH/harness" "$TASK_SCRATCH/runtime"
tar -xf "$TASK_ROOT/source.tar" -C "$TASK_SCRATCH/source"
python3 "$TASK_SCRATCH/source/scripts/flash_mixed_cosine120/stage_checks.py" "$TASK_ROOT"
tar -xzf "$TASK_ROOT/harness-source.tar.gz" -C "$TASK_SCRATCH/harness"
tar --zstd -xf "$TASK_ROOT/harness-runtime.tar.zst" -C "$TASK_SCRATCH/runtime"
python3 "$TASK_SCRATCH/source/scripts/flash_mixed_cosine120/bootstrap.py" "$TASK_SCRATCH/runtime" "$TASK_SCRATCH/harness" "$TASK_SCRATCH/source/scripts/flash_mixed_cosine120/artifacts.json"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$TASK_SCRATCH/source:$TASK_SCRATCH/source/scripts/flash_mixed_cosine120/conversion:$TASK_ROOT/test-deps"
export FLASH_MIXED_HARNESS_IDENTITY="$TASK_SCRATCH/runtime/HARNESS_IMPORTED.json"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 GLOO_SOCKET_IFNAME=lo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TRITON_CACHE_DIR="$TASK_SCRATCH/triton" TORCHINDUCTOR_CACHE_DIR="$TASK_SCRATCH/inductor"
cd "$TASK_SCRATCH/source"
"$TASK_SCRATCH/runtime/bin/python" -m pytest tests/test_flash_sft.py tests/test_flash_sft_logprobs.py tests/test_flash_sft_logprobs_sharded.py tests/test_flash_sft_cosine120.py tests/test_kimi_conversion_diagnostics.py tests/test_mixed_sft_provenance.py tests/logprob -q --tb=short
"$TASK_SCRATCH/runtime/bin/python" scripts/flash_mixed_cosine120/official_eval.py --checkpoint /data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307 --run-dir /data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/evaluation-contract-check --receipt /data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/OFFICIAL_EVALUATION_BUILD_PASSED.json
case "${MIXED_PREPARATION_MODE:-build}" in
  build) MODE=() ;;
  build-only) MODE=(--build-only) ;;
  verify-only) MODE=(--verify-only) ;;
  *) exit 2 ;;
esac
"$TASK_SCRATCH/runtime/bin/python" scripts/flash_mixed_cosine120/prepare.py "${MODE[@]}"
