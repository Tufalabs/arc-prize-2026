#!/bin/bash
#SBATCH --job-name=in-job
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-in-job-%j.out
# Run a recipe command from arc-training, in its dedicated drafter environment.
# ARC2 and ARC3 share the trainer; DRAFTER_TARGET selects the teacher.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from arc-training}"
ulimit -Sn "$(ulimit -Hn)"
unset VIRTUAL_ENV
cache_root="${DRAFTER_CACHE_ROOT:-/cache/users/$(id -u)/drafter}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
source spec_decode/tools/environment.sh
setup_drafter_environment "$(pwd)/spec_decode" "${cache_root}"
export PYTHONPATH="$(pwd)/spec_decode${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
exec "$@"
