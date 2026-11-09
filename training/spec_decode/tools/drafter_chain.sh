#!/bin/bash
# A drafter for a new target, from that target's own long runs: each run's
# chains as a dataset (two whole games held out), its extracts (SHARDS B200
# jobs per run plus one for the held-out games), then training on the caches
# once every extract has succeeded (drafter.jobs --after). Run it from this
# checkout with the target and its drafter base in the environment:
#   DRAFTER_TARGET=T DRAFTER_BASE=B sbatch spec_decode/tools/in_job.sbatch bash spec_decode/tools/drafter_chain.sh NAME SHARDS RUN [RUN ...]
# The datasets are NAME-0, NAME-1, ... (one per RUN) and the run is runs/NAME;
# spec_decode/tools/drafter_finish.sh turns its last checkpoint into a served head.
set -euo pipefail
name="$1" shards="$2"
shift 2
: "${DRAFTER_TARGET:?the target the head drafts for}" "${DRAFTER_BASE:?its drafter base (python -m drafter.extract_base)}"
: "${DRAFTER_SERVING_PYTHON:?qualified serving interpreter for target extraction}"
D="${DRAFTER_ROOT:-/data/shared/research/arc_3_results/$(id -un)/drafter}"
job() {
  local id
  id="$(python -m drafter.jobs "$@" | sed -n 's/^submitted \([0-9]*\)$/\1/p')"
  [ -n "${id}" ] || { echo "no job id from drafter.jobs $1" >&2; return 1; }
  echo "${id}"
}
extracts=() train=() val=()
index=0
for run in "$@"; do
  ds="${name}-${index}"
  python -m drafter.chains build "$D/datasets/${ds}" "${run}" --val-games 2 --seed 0
  for ((shard = 0; shard < shards; shard++)); do
    id="$(job "${ds}-extract-${shard}" --minutes 150 -- \
      drafter.extract "$D/datasets/${ds}/train" "$D/caches/${ds}" --shard "${shard}/${shards}")"
    extracts+=("${id}")
  done
  id="$(job "${ds}-extract-val" --minutes 90 -- drafter.extract "$D/datasets/${ds}/val" "$D/caches/${ds}-val")"
  extracts+=("${id}")
  train+=(--train "$D/caches/${ds}:$D/datasets/${ds}/train")
  val+=(--val "$D/caches/${ds}-val:$D/datasets/${ds}/val")
  index=$((index + 1))
done
after="$(IFS=:; echo "${extracts[*]}")"
echo "extracts ${after}"
# The dedicated distillation recipe at the cells' k=4.
id="$(job "${name}-train" --gpus 4 --minutes 330 --after "${after}" -- drafter.train "$D/runs/${name}" \
  "${train[@]}" "${val[@]}" --gpus 4 --steps 4 --lr 2e-5 --epochs 1 --save-minutes 30 --val-every 50 --minutes 300)"
echo "train ${id}"
