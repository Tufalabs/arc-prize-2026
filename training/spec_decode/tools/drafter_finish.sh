#!/bin/bash
# A trained drafter (spec_decode/tools/drafter_chain.sh NAME) to its served head in OUT:
# the last checkpoint exported over the drafter base, its routed experts in
# NVFP4 (a B200 job calibrated on the held-out games at k=4), then the 16k
# draft vocabulary ranked on the training chains' generated tokens.
#   DRAFTER_TARGET=T DRAFTER_BASE=B sbatch spec_decode/tools/in_job.sbatch bash spec_decode/tools/drafter_finish.sh NAME OUT
set -euo pipefail
name="$1" out="$2"
: "${DRAFTER_TARGET:?the target the head drafts for}" "${DRAFTER_BASE:?its drafter base (python -m drafter.extract_base)}"
D="${DRAFTER_ROOT:-/data/shared/research/arc_3_results/$(id -un)/drafter}"
if [ -e "${out}" ]; then
  echo "${out} exists" >&2
  exit 1
fi
last="$(find "$D/runs/${name}/checkpoints" -maxdepth 1 -name 'update-*' | sort | tail -1)"
[ -n "${last}" ] || { echo "no checkpoint under $D/runs/${name}" >&2; exit 1; }
echo "exporting ${last}"
python -m drafter.export "${last}/head.safetensors" "$D/heads/${name}"
calibrate=()
for ds in "$D/datasets/${name}"-[0-9]*; do
  calibrate+=(--calibrate "$(basename "${ds}")=$D/caches/$(basename "${ds}")-val:${ds}/val")
done
nvfp4="$(python -m drafter.jobs "${name}-nvfp4" --minutes 60 -- \
  drafter.nvfp4 "$D/heads/${name}" "$D/heads/${name}-nvfp4" "${calibrate[@]}" --steps 4 |
  sed -n 's/^submitted \([0-9]*\)$/\1/p')"
[ -n "${nvfp4}" ] || { echo "no job id for the NVFP4 step" >&2; exit 1; }
echo "nvfp4 ${nvfp4}"
vocab="$(python -m drafter.jobs "${name}-vocab" --gpus 0 --minutes 90 --after "${nvfp4}" -- \
  drafter.reduce_vocab --src "$D/heads/${name}-nvfp4" --size 16384 \
  --chains "$D/datasets/${name}"-[0-9]*/train/chains.jsonl \
  --heldout "$D/datasets/${name}"-[0-9]*/val/chains.jsonl --out "${out}" |
  sed -n 's/^submitted \([0-9]*\)$/\1/p')"
[ -n "${vocab}" ] || { echo "no job id for vocabulary reduction" >&2; exit 1; }
echo "vocab ${vocab}"
