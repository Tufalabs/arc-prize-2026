#!/bin/bash
#SBATCH --job-name=flash-prune-plan
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out
# CPU-only: REAP-307 plans from a replay's statistics, once every calibration
# rollout is in. Submit from the arc-training checkout:
#   sbatch scripts/prune_plan.sbatch <stats dir> <plans dir> <rollouts expected> <reference checkpoint dir>...
set -euo pipefail
. /etc/kander/job-env.sh
stats="$1" plans="$2" need="$3"
shift 3
cd "${SLURM_SUBMIT_DIR:?submit from the checkout}"
export ARC3_FLASH_REPO_ROOT="${PWD}"
source scripts/flash_env.sh
n="$(find "${stats}" -maxdepth 1 -name '*.npz' ! -name '*.tmp.npz' | wc -l)"
echo "${n} replayed rollouts in ${stats}"
if [ "${n}" -lt "${need}" ]; then
  echo "fewer than the ${need} calibration rollouts" >&2
  exit 1
fi
"${ARC3_FLASH_PYTHON}" scripts/prune_plans.py --stats "${stats}" --out "${plans}" --keep 307 --reference "$@"
ls -l "${plans}"
