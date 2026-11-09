#!/bin/bash
#SBATCH --job-name=mixed120-cekl-cosine
#SBATCH --account=arc
#SBATCH --partition=arc
#SBATCH --qos=arc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:b200:4
#SBATCH --mem-per-gpu=239G
#SBATCH --time=24:00:00
#SBATCH --output=/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/training-job-%j.log
set -euo pipefail
# Proportional host RAM: four B200s reserve 956 GiB; eight reserve 1912 GiB.
python3 -u /data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/artifacts/train.py
