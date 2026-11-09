#!/bin/bash
#SBATCH --job-name=arc3-flash-dev
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=160G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
# One-GPU development job inside the Flash runtime. Usage:
#   ARC3_RUN_DIR=<fresh dir> sbatch scripts/flash_dev.sbatch scripts/<probe>.py --args
# The trailing arguments are handed to the Flash interpreter verbatim.
set -euo pipefail
. /etc/kander/job-env.sh
REPO_ROOT="${ARC3_JOB_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit from the checkout}}"
cd "${REPO_ROOT}"
: "${ARC3_RUN_DIR:?set a fresh output directory}"
mkdir -p "${ARC3_RUN_DIR}"
export ARC3_FLASH_REPO_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/scripts/flash_env.sh"
export OMP_NUM_THREADS=8
"${ARC3_FLASH_PYTHON}" "${REPO_ROOT}/scripts/flash_runtime.py" attest \
  --repo "${REPO_ROOT}" --output "${ARC3_RUN_DIR}/runtime_attestation.json"
echo "flash dev job ${SLURM_JOB_ID}: python=${ARC3_FLASH_PYTHON} run_dir=${ARC3_RUN_DIR} args=$*"
exec "${ARC3_FLASH_PYTHON}" "$@"
