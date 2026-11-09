#!/bin/bash
#SBATCH --job-name=flash-prune-physical
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# CPU-only: cut the routed experts a plan drops out of the 512-expert
# Flash-Next NVFP4 checkpoint (scripts/prune_flashnext_experts.py), on the Flash runtime's torch + safetensors.
# Usage:
#   sbatch scripts/prune_physical.sbatch <plan.json> <out dir> [<plan.json> <out dir> ...]
# Refuses an existing output directory; ARC3_PRUNE_SOURCE overrides the source.
set -euo pipefail
. /etc/kander/job-env.sh
REPO_ROOT="${ARC3_JOB_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit from the checkout}}"
cd "${REPO_ROOT}"
SOURCE="${ARC3_PRUNE_SOURCE:-/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4}"
if [ $(( $# % 2 )) -ne 0 ] || [ $# -eq 0 ]; then
  echo "usage: prune_physical.sbatch <plan.json> <out dir> [...]" >&2
  exit 2
fi
export ARC3_FLASH_REPO_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/scripts/flash_env.sh"
while [ $# -gt 0 ]; do
  plan="$1" out="$2"
  shift 2
  if [ -e "${out}" ]; then
    echo "FATAL: ${out} exists; prune into a fresh directory" >&2
    exit 1
  fi
  echo "prune job ${SLURM_JOB_ID}: ${SOURCE} x ${plan} -> ${out}"
  "${ARC3_FLASH_PYTHON}" scripts/prune_flashnext_experts.py \
    --src "${SOURCE}" --plan "${plan}" --out "${out}"
done
