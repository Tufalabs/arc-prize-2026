#!/bin/bash
#SBATCH --job-name=toolcalls-cekl
#SBATCH --account=arc
#SBATCH --partition=arc
#SBATCH --qos=arc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:b200:4
#SBATCH --mem-per-gpu=239G
#SBATCH --time=24:00:00
#SBATCH --output=/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-toolcalls-20261030/experiments/%x-job-%j.log
# Submit with: sbatch --export=ALL,TOOLCALLS_EXPERIMENT=<config stem> --job-name=<config stem> \
#   --output=.../experiments/<config stem>/training-job-%j.log train.sbatch
set -euo pipefail
: "${TOOLCALLS_EXPERIMENT:?config stem under configs/flash_sft, e.g. reap307-kimi-mixed-toolcalls}"
python3 -u /data/shared/research/arc_3_results/shared/reap307-kimi-mixed-toolcalls-20261030/artifacts/train.py
