#!/usr/bin/env bash
# Two-job Slurm chain, fully detached from the submitting shell:
#   (1) CPU pre-pack job builds the dataset packs into dataset_cache
#   (2) GPU training job starts only after (1) succeeds (--dependency=afterok)
# and finds the packs, so it spends its whole time limit training.
#
#   ./slurm/submit_sft_chain.sh configs/sft.json
#   SFT_TRAIN_TIME=12:00:00 ./slurm/submit_sft_chain.sh <config>   # default 12h
#   SFT_GPUS=4 ./slurm/submit_sft_chain.sh <config>   # DDP via torchrun; default 1
#   SFT_EXTRA_ARGS=--extra=sft-rearc ./slurm/submit_sft_chain.sh <config>
#   SFT_PREPACK_DEPENDENCY=afterany:<jobid> ./slurm/submit_sft_chain.sh <config>
#       # hold the chain until <jobid> ends -- e.g. the run still generating
#       # the traces this config selects from
#
# If the prepack fails, the training job stays pending as
# DependencyNeverSatisfied -- scancel it after fixing the cause.
set -euo pipefail

# Worktree support: resolve the checkout this script actually lives in rather
# than a fixed path, so `git worktree` checkouts and second clones submit their
# OWN code. A hardcoded path fails silently -- the job runs the other
# checkout's code and reports success. SFT_PROJECT_DIR still overrides.
PROJECT_DIR="${SFT_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SFT_CONFIG="${SFT_CONFIG:-${1:-${PROJECT_DIR}/configs/sft.json}}"
SFT_CONFIG="$(readlink -f "${SFT_CONFIG}")"
TRAIN_TIME="${SFT_TRAIN_TIME:-12:00:00}"
CONFIG_GPUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("gpus") or 1)' "${SFT_CONFIG}")"
TRAIN_GPUS="${SFT_GPUS:-${CONFIG_GPUS}}"

RUN_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name") or "sft")' "${SFT_CONFIG}")"

# Optional gate: hold the prepack until another job finishes.
# This must be an explicit --dependency flag: kander's sbatch silently
# ignores the SBATCH_DEPENDENCY environment variable. Measured against the
# same anchor job -- with the env var the job started immediately and
# reported Dependency=(null); with the flag it stayed PENDING as
# afterany:<jobid>(unfulfilled). Anything here relying on SBATCH_* env
# inputs is worth re-checking for the same reason.
PREPACK_DEP=()
if [ -n "${SFT_PREPACK_DEPENDENCY:-}" ]; then
  PREPACK_DEP=(--dependency="${SFT_PREPACK_DEPENDENCY}")
  echo ">> prepack gated on ${SFT_PREPACK_DEPENDENCY}"
fi
PREPACK_ID="$(sbatch --parsable --job-name="${RUN_NAME}-prepack" \
  "${PREPACK_DEP[@]}" \
  --chdir="${PROJECT_DIR}" \
  --export=ALL,SFT_CONFIG="${SFT_CONFIG}",SFT_PROJECT_DIR="${PROJECT_DIR}" \
  --output="${PROJECT_DIR}/slurm/slurm-%x-%j.out" \
  "${PROJECT_DIR}/slurm/sft_prepack.sbatch")"
echo ">> prepack job ${PREPACK_ID} (${RUN_NAME}-prepack)"

TRAIN_ID="$(sbatch --parsable --job-name="${RUN_NAME}" --time="${TRAIN_TIME}" \
  --gpus="b200:${TRAIN_GPUS}" \
  --dependency=afterok:"${PREPACK_ID}" \
  --chdir="${PROJECT_DIR}" \
  --export=ALL,SFT_CONFIG="${SFT_CONFIG}",SFT_PROJECT_DIR="${PROJECT_DIR}" \
  --output="${PROJECT_DIR}/slurm/slurm-%x-%j.out" \
  "${PROJECT_DIR}/slurm/sft_train.sbatch")"
echo ">> training job ${TRAIN_ID} (${RUN_NAME}, time=${TRAIN_TIME}, after ${PREPACK_ID})"
