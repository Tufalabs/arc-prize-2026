#!/bin/bash
# Build the three Flash RL job environments (trainer, harness, arc2-worker) for
# one revision: submits scripts/build_env.sbatch once per recipe that has no
# matching artifact yet (docs/environments.md). Run it on the login node:
#
#   scripts/build_envs.sh [--revision REV] [--out DIR] [--repo DIR] [--force | --check] [-- SBATCH_OPTION...]
#
#   --revision REV  the commit to build (default: HEAD of --repo). It must be a
#                   commit in --repo; it need not be pushed, since each job
#                   archives it from that clone when it starts (do not prune it
#                   before then). The commits it pins for the `platypus` and
#                   `arc2-platypus` submodules must be on their remotes: the jobs
#                   clone them. Push the revision too if others should rebuild
#                   from the provenance the manifests record.
#   --out DIR       artifact root on /data (default: $ARC_TRAINING_ENVS, else
#                   /data/personal/$USER/arc-training-envs); created if missing.
#                   Build logs go there as slurm-<job>.out.
#   --repo DIR      the clone to build from (default: the checkout holding this script)
#   --force         submit every recipe, even one whose artifact already matches
#   --check         submit nothing: report which recipes have an artifact for the
#                   revision (and, for one that has none, the nearest artifact and
#                   the fields that differ); exit 3 if any has none
#   -- OPTIONS      passed to sbatch, e.g. -- --exclude=dgx02
#
# A recipe whose artifact already matches the revision (scripts/resolve_envs.py:
# same locks, submodule pins and interpreter) is reported, not submitted. A
# submitted build whose key exists only records a receipt (build_env.sbatch never
# overwrites an artifact). Builds are CPU jobs (--qos=dev, three hours at most);
# scripts/flash_rl_submit.sh then finds the artifacts in the same root.
set -euo pipefail

die() { echo "build_envs: $*" >&2; exit 2; }
usage() { sed -n '2,/^set -euo pipefail/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME=${USER:-$(id -un)}
REVISION=HEAD REPO="" FORCE=0 CHECK=0 SBATCH_EXTRA=()
OUT="${ARC_TRAINING_ENVS:-/data/personal/$USER_NAME/arc-training-envs}"
while (( $# )); do
  case "$1" in
    --revision) REVISION=${2:?--revision needs a value}; shift 2 ;;
    --out) OUT=${2:?--out needs a value}; shift 2 ;;
    --repo) REPO=${2:?--repo needs a value}; shift 2 ;;
    --force) FORCE=1; shift ;;
    --check) CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; SBATCH_EXTRA=("$@"); break ;;
    *) usage >&2; die "unknown option $1" ;;
  esac
done
OUT=${OUT%/}
(( ! (FORCE && CHECK) )) || die "--force builds and --check only reports: pass one of them"
if [[ -z "$REPO" ]]; then
  REPO=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null) && [[ "$REPO/scripts" -ef "$HERE" ]] ||
    die "$HERE is not at the root of a training repository; pass --repo DIR (see docs/environments.md#building)"
fi
REPO=$(cd "$REPO" && pwd)
REV=$(git -C "$REPO" rev-parse --verify --quiet "$REVISION^{commit}") ||
  die "$REVISION is not a commit in $REPO (commit it, or fetch it into --repo)"
git -C "$REPO" cat-file -e "$REV:scripts/build_env.sbatch" 2>/dev/null ||
  die "$REV has no scripts/build_env.sbatch: build from a revision that carries the environment tooling (see docs/environments.md#building)"
RECIPES=(trainer harness arc2-worker)
# Plan every recipe first (revision, submodule pins, --out), so a bad request submits nothing.
for recipe in "${RECIPES[@]}"; do
  bash "$HERE/build_env.sbatch" "$recipe" "$REV" --repo "$REPO" --out "$OUT" --print-plan > /dev/null ||
    die "cannot plan $recipe at $REV (above)"
done
# The next step: flash_rl_submit.sh runs its own checkout's HEAD unless given -- <repo> <revision>.
SUBMIT="$HERE/flash_rl_submit.sh --config configs/flash_8gpu_arc2_grpo_tokmean.json"
[[ "$OUT" == "${ARC_TRAINING_ENVS:-/data/personal/$USER_NAME/arc-training-envs}" ]] || SUBMIT+=" --envs $OUT"
TOP=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null) || TOP=""
if [[ -z "$TOP" || ! "$REPO" -ef "$TOP" || "$REV" != "$(git -C "$TOP" rev-parse --verify --quiet 'HEAD^{commit}')" ]]; then
  SUBMIT+=" -- $(printf '%q' "$REPO") $REV"
fi
SMOKE="Or, for a smoke run first, derive a short config as training/FLASH_RL.md shows (\"The recipe and"
SMOKE+=" smoke configs\") and pass its absolute path to --config."

if (( CHECK )); then
  # Read-only: the resolver names each missing recipe's nearest artifact on stderr.
  STATUS=0
  RESOLVED=$(python3 "$HERE/resolve_envs.py" --root "$OUT" --repo "$REPO" --revision "$REV" \
    "${RECIPES[@]}") || STATUS=$?
  [[ $STATUS == 0 || $STATUS == 3 ]] || die "scripts/resolve_envs.py failed (above)"
  echo "build_envs: revision $REV of $REPO; artifacts in $OUT"
  MISSING=()
  for recipe in "${RECIPES[@]}"; do
    tarball=$(awk -v recipe="$recipe" '$1 == recipe {print $2}' <<< "$RESOLVED")
    if [[ -n "$tarball" ]]; then
      echo "build_envs: $recipe is built: $tarball"
    else
      echo "build_envs: $recipe is not built (resolve_envs above names the nearest artifact)"
      MISSING+=("$recipe")
    fi
  done
  if (( ${#MISSING[@]} )); then
    echo "build_envs: not built: ${MISSING[*]}; the same command without --check submits their builds"
    exit 3
  fi
  echo "build_envs: all three are built. Submit the full recipe: $SUBMIT"
  echo "$SMOKE"
  exit 0
fi

mkdir -p "$OUT"
RESOLVED=""
if (( ! FORCE )); then
  RESOLVED=$(python3 "$HERE/resolve_envs.py" --root "$OUT" --repo "$REPO" --revision "$REV" --quiet \
    "${RECIPES[@]}") || [[ $? == 3 ]] || die "scripts/resolve_envs.py failed (above)"
fi
echo "build_envs: revision $REV of $REPO; artifacts in $OUT"
JOBS=() LOGS=()
for recipe in "${RECIPES[@]}"; do
  tarball=$(awk -v recipe="$recipe" '$1 == recipe {print $2}' <<< "$RESOLVED")
  if [[ -n "$tarball" ]]; then
    echo "build_envs: $recipe is up to date: $tarball"
    continue
  fi
  # The revision's own builder, so the manifest records builder_matches_revision=true.
  submitted=$(sbatch --parsable --job-name="build-env-$recipe" --output="$OUT/slurm-%j.out" \
    ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"} <(git -C "$REPO" show "$REV:scripts/build_env.sbatch") \
    "$recipe" "$REV" --repo "$REPO" --out "$OUT")
  JOBS+=("${submitted%%;*}")
  LOGS+=("$OUT/slurm-${submitted%%;*}.out")
  echo "build_envs: $recipe: job ${submitted%%;*}"
done
if (( ${#JOBS[@]} == 0 )); then
  echo "build_envs: nothing to build. Submit the full recipe: $SUBMIT"
  echo "$SMOKE"
  exit 0
fi
IDS=$(IFS=,; echo "${JOBS[*]}")
echo "build_envs: jobs $IDS"
echo "Check progress: sacct -X -j $IDS -o JobID,JobName%22,State,Elapsed"
echo "Results (one BUILD_ENV_RESULT line per finished build): grep -h BUILD_ENV_RESULT ${LOGS[*]}"
echo "Once all have COMPLETED, submit the full recipe: $SUBMIT"
echo "$SMOKE"
