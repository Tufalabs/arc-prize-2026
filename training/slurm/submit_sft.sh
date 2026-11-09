#!/usr/bin/env bash
# Submit the SFT job with the Slurm job name taken from the config run_name, so
# both `squeue` and the log filename (slurm-<run_name>-<jobid>.out) reflect it.
#
#   ./slurm/submit_sft.sh                         # default configs/sft.json
#   ./slurm/submit_sft.sh configs/sft.json
#   SFT_CONFIG=configs/sft.json ./slurm/submit_sft.sh
#   SFT_EXTRA_ARGS=--extra=sft-rearc ./slurm/submit_sft.sh <config>
set -euo pipefail

# Worktree support: resolve the checkout this script actually lives in rather
# than a fixed path, so `git worktree` checkouts and second clones submit their
# OWN code. A hardcoded path fails silently -- the job runs the other
# checkout's code and reports success. SFT_PROJECT_DIR still overrides.
PROJECT_DIR="${SFT_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SFT_CONFIG="${SFT_CONFIG:-${1:-${PROJECT_DIR}/configs/sft.json}}"
SFT_UV_EXTRA_ARGS=()
if [ -n "${SFT_EXTRA_ARGS:-}" ]; then
  read -r -a SFT_UV_EXTRA_ARGS <<< "${SFT_EXTRA_ARGS}"
fi

RUN_NAME="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(\"run_name\") or \"sft\")" "${SFT_CONFIG}")"

# Pre-pack the dataset cache on the LOGIN node before submitting. The compute
# nodes share one NFS client that serializes small-file reads (~40 files/s
# under neighbor load vs ~2000/s serial here), so an in-job pack build from
# raw traces is the slowest phase of the whole pipeline by far. --check-data
# selects + builds + packs WITHOUT loading the model; when the packs already
# exist this is a no-op that takes seconds. SFT_STAGE_DIR is unset so the
# build reads the raw traces straight from /data (fast from here). The venv
# goes to login-local /scratch, never the repo .venv on /data NFS.
if [ "${SFT_SKIP_PREPACK:-0}" != "1" ]; then
  echo ">> pre-packing dataset cache on $(hostname) (SFT_SKIP_PREPACK=1 to skip)"
  PREPACK_VENV="/scratch/${USER}/venvs/arc3-training-sft"
  mkdir -p "$(dirname "${PREPACK_VENV}")" "/scratch/${USER}/sft_build"
  (cd "${PROJECT_DIR}" && env -u SFT_STAGE_DIR -u VIRTUAL_ENV \
    UV_PROJECT_ENVIRONMENT="${PREPACK_VENV}" \
    SFT_BUILD_WORK_DIR="/scratch/${USER}/sft_build" \
    uv run --locked --extra sft "${SFT_UV_EXTRA_ARGS[@]}" \
      python -m sft.train --config "${SFT_CONFIG}" --check-data)
fi

echo ">> submitting run_name=${RUN_NAME}  config=${SFT_CONFIG}"
exec sbatch --job-name="${RUN_NAME}" \
  --chdir="${PROJECT_DIR}" \
  --export=ALL,SFT_CONFIG="${SFT_CONFIG}",SFT_PROJECT_DIR="${PROJECT_DIR}" \
  --output="${PROJECT_DIR}/slurm/slurm-%x-%j.out" \
  "${PROJECT_DIR}/slurm/sft_train.sbatch"
