#!/bin/bash
#SBATCH --job-name=toolcalls-targets
#SBATCH --account=arc
#SBATCH --partition=arc
#SBATCH --qos=arc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-toolcalls-20261030/slurm-%j.log
# Rebuild the recorded Kimi logprob targets of PR 121's frozen mixed corpus with
# this branch's alignment scope (reasoning, content, tool calls, control-token
# decisions). Inputs are exactly the baseline's deduplicated rows, sources and
# tokenizer; row order and every baseline field are preserved, only
# `teacher_logprob_targets` changes. The submitter stages `sft/__init__.py` and
# `sft/logprob/` into $OUT/snapshot and records the commit in SNAPSHOT.json.
set -euo pipefail
. /etc/kander/job-env.sh
OUT=${TARGETS_OUT:-/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-toolcalls-20261030}
BASELINE=${TARGETS_BASELINE:-/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030}
ROWS="$BASELINE/deduplicated-reviewed19"
TOKENIZER=${TARGETS_TOKENIZER:-/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307/tokenizer.json}
SNAPSHOT="$OUT/snapshot"
WORKDIR="$SCRATCH_DIR/targets"
cleanup() { [[ "$WORKDIR" == /scratch/* ]] && rm -rf "$WORKDIR"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
mkdir -p "$WORKDIR/traces" "$WORKDIR/out" "$OUT/data"
# Stage only the step files of the baseline's 19 raw collections; media is not needed.
python3 - "$BASELINE/sources.json" "$WORKDIR" <<'PY'
import json, sys
from pathlib import Path
roots = json.loads(Path(sys.argv[1]).read_text())
staged = {name: f"{sys.argv[2]}/traces/{name}" for name in roots}
Path(sys.argv[2], "sources.json").write_text(json.dumps(staged, indent=1) + "\n")
Path(sys.argv[2], "roots.tsv").write_text("".join(f"{name}\t{root}\n" for name, root in roots.items()))
PY
while IFS=$'\t' read -r name root; do
  rsync -a --include='*/' --include='steps.jsonl' --exclude='*' "$root/" "$WORKDIR/traces/$name/"
done < "$WORKDIR/roots.tsv"
cp "$TOKENIZER" "$WORKDIR/tokenizer.json"
cd "$SNAPSHOT"
for split in rows validation-probe; do
  python3 -m sft.logprob.targets --rows "$ROWS/$split.jsonl" --sources "$WORKDIR/sources.json" \
    --allow-missing-logprobs --tokenizer "$WORKDIR/tokenizer.json" --out "$WORKDIR/out/$split.jsonl"
done
rsync -a "$WORKDIR/out/" "$OUT/data/"
python3 - "$OUT" "$BASELINE" "$SNAPSHOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, baseline, snapshot = map(Path, sys.argv[1:4])
FIELD = "teacher_logprob_targets"
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
def stripped(path):
    """SHA256 of the rows with the targets removed: the baseline fields in order."""
    h = hashlib.sha256(); n = 0
    with path.open() as f:
        for line in f:
            row = json.loads(line); row.pop(FIELD, None); n += 1
            h.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    return h.hexdigest(), n
ready = {"status": "ready", "baseline": str(baseline), "snapshot": json.loads((snapshot / "SNAPSHOT.json").read_text())}
for split in ("rows", "validation-probe"):
    mine, theirs = out / "data" / f"{split}.jsonl", baseline / "data-reviewed19" / f"{split}.jsonl"
    report = json.loads((out / "data" / f"{split}.jsonl.report.json").read_text())
    their_report = json.loads((baseline / "data-reviewed19" / f"{split}.jsonl.report.json").read_text())
    if report["input_rows_sha256"] != their_report["input_rows_sha256"]:
        raise SystemExit(f"{split}: input rows differ from the baseline's targets input")
    (digest, n), (their_digest, their_n) = stripped(mine), stripped(theirs)
    if (digest, n) != (their_digest, their_n):
        raise SystemExit(f"{split}: baseline fields differ from {theirs}")
    ready[split] = {"data": str(mine), "data_sha256": sha(mine), "rows": n,
                    "baseline_data_sha256": sha(theirs), "baseline_fields_sha256": digest,
                    "input_rows_sha256": report["input_rows_sha256"],
                    "counts": report["counts"], "baseline_counts": their_report["counts"], "by_source": report["by_source"]}
    ready["tokenizer_sha256"] = report["tokenizer_sha256"]
    ready["scope"] = report["scope"]
(out / "data" / "DATA_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
print(json.dumps({k: {"rows": v["rows"], "data_sha256": v["data_sha256"], "aligned": v["counts"]["aligned_positions"],
                      "baseline_aligned": v["baseline_counts"]["aligned_positions"], "completion_tokens": v["counts"]["completion_tokens"]}
                  for k, v in ready.items() if isinstance(v, dict) and "counts" in v}, indent=1))
PY
