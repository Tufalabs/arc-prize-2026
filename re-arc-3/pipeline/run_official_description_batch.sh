#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_WORKERS="${MAX_WORKERS:-1}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/codex-batch/official-game-descriptions}"
EXCLUDE_GAMES="${EXCLUDE_GAMES:-ls20 ar25 ar24}"
if [[ "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_PATH="$OUTPUT_DIR"
else
  OUTPUT_PATH="$ROOT/$OUTPUT_DIR"
fi

if command -v pipeline-codex-batch >/dev/null 2>&1; then
  RUNNER=(pipeline-codex-batch)
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  RUNNER=("$ROOT/.venv/bin/python" -m pipeline.codex_batch)
else
  RUNNER=(python -m pipeline.codex_batch)
fi

JOBS_FILE="$ROOT/pipeline/jobs/official_game_descriptions.csv"
FILTERED_JOBS="$(mktemp --suffix=.csv)"
cleanup() {
  rm -f "$FILTERED_JOBS"
}
trap cleanup EXIT

python - "$JOBS_FILE" "$FILTERED_JOBS" "$EXCLUDE_GAMES" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
excluded = set(sys.argv[3].split())

with source.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

if not rows:
    raise SystemExit(f"No jobs found in {source}")

kept = [row for row in rows if row.get("game", "") not in excluded]

with target.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(kept)

print(f"Running {len(kept)} jobs; excluded: {', '.join(sorted(excluded))}")
PY

"${RUNNER[@]}" \
  --jobs "$FILTERED_JOBS" \
  --prompt-file "$ROOT/pipeline/prompts/official_game_description_prompt.md" \
  --base-dir "$ROOT" \
  --reasoning-effort "$REASONING_EFFORT" \
  --max-workers "$MAX_WORKERS" \
  --continue-on-error \
  --output-dir "$OUTPUT_PATH"
