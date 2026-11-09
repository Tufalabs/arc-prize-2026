#!/bin/bash
#SBATCH --job-name=kimi-logprob-targets
#SBATCH --account=arc
#SBATCH --partition=arc
#SBATCH --qos=arc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/shared/research/arc_3_results/shared/reap307-kimi-logprob-toolcalls-20261030/slurm-%j.log
# Rebuild the recorded Kimi logprob targets (reasoning, content and tool-call
# values) for the four row files of the running CE+KL experiment. Row order and
# every baseline field are preserved; only `teacher_logprob_targets` changes.
# The submitter stages `sft/__init__.py` and `sft/logprob/` into $OUT/snapshot
# and records the commit in $OUT/snapshot/SNAPSHOT.json. Stdlib only.
set -euo pipefail
. /etc/kander/job-env.sh
OUT=${TARGETS_OUT:-/data/shared/research/arc_3_results/shared/reap307-kimi-logprob-toolcalls-20261030}
ROWS=${TARGETS_ROWS:-/data/shared/research/arc_3_results/shared/reap307-kimi-logprob-only-20261029/data}
TRACES=${TARGETS_TRACES:-/data/shared/research/arc_3_results/shared/kimi-sft-baseline-20261029/traces}
TOKENIZER=${TARGETS_TOKENIZER:-/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307/tokenizer.json}
SNAPSHOT="$OUT/snapshot"
WORKDIR="$SCRATCH_DIR/targets"
cleanup() { [[ "$WORKDIR" == /scratch/* ]] && rm -rf "$WORKDIR"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
mkdir -p "$WORKDIR/traces" "$WORKDIR/rows" "$WORKDIR/out" "$OUT/data"
# Stage only the step files of the two source sets; media is not needed.
for source in rearc-rest arc3-hard-r2; do
  rsync -a --include='*/' --include='steps.jsonl' --exclude='*' "$TRACES/$source/" "$WORKDIR/traces/$source/"
done
printf '{"rearc-rest": "%s/traces/rearc-rest", "arc3-hard-r2": "%s/traces/arc3-hard-r2"}\n' \
  "$WORKDIR" "$WORKDIR" > "$WORKDIR/sources.json"
cp "$TOKENIZER" "$WORKDIR/tokenizer.json"
cd "$SNAPSHOT"
for split in train validation probe qualification; do
  python3 - "$ROWS/$split.jsonl" "$WORKDIR/rows/$split.jsonl" <<'PY'
import json, sys
from sft.logprob.targets import FIELD
with open(sys.argv[1]) as src, open(sys.argv[2], "w") as dst:
    for line in src:
        row = json.loads(line)
        row.pop(FIELD, None)
        dst.write(json.dumps(row, separators=(",", ":")) + "\n")
PY
  python3 -m sft.logprob.targets --rows "$WORKDIR/rows/$split.jsonl" --sources "$WORKDIR/sources.json" \
    --tokenizer "$WORKDIR/tokenizer.json" --out "$WORKDIR/out/$split.jsonl"
done
rsync -a "$WORKDIR/out/" "$OUT/data/"
python3 - "$OUT" "$ROWS" "$TRACES" "$SNAPSHOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, rows, traces, snapshot = map(Path, sys.argv[1:5])
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
ready = {"status": "ready", "input_rows": str(rows), "traces": str(traces),
         "snapshot": json.loads((snapshot / "SNAPSHOT.json").read_text())}
for split in ("train", "validation", "probe", "qualification"):
    report = json.loads((out / "data" / f"{split}.jsonl.report.json").read_text())
    ready[split] = {"data": str(out / "data" / f"{split}.jsonl"), "data_sha256": sha(out / "data" / f"{split}.jsonl"),
                    "input_sha256": sha(rows / f"{split}.jsonl"), "counts": report["counts"], "by_source": report["by_source"]}
    ready["tokenizer_sha256"] = report["tokenizer_sha256"]
    ready["scope"] = report["scope"]
(out / "data" / "DATA_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
print(json.dumps({k: v["counts"] for k, v in ready.items() if isinstance(v, dict) and "counts" in v}, indent=1))
PY
