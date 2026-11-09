#!/bin/bash
# Compose a Flash learner allocation plus identical groups of rollout nodes and
# submit scripts/flash_rl.sbatch (training/FLASH_RL.md). Usually only the config:
#
#   scripts/flash_rl_submit.sh --config configs/flash_8gpu_arc2_grpo_tokmean.json
#
# runs HEAD of the checkout holding this script with the environments built for
# it (scripts/build_envs.sh) in a new run directory. Options:
#   --config FILE        the recipe (or ARC3_CONFIG); a relative path is read from
#                        the revision the job archives
#   --run-dir DIR        run directory under /data (or ARC3_RUN_DIR); reuse it to
#                        resume. Default: /data/personal/$USER/arc-training-runs/
#                        <config stem>-<YYYYMMDD-HHMM>/lora
#   --envs DIR           artifact root the environments are resolved from (default
#                        $ARC_TRAINING_ENVS, else /data/personal/$USER/arc-training-envs)
#   --allow-dirty        submit HEAD although the working tree has uncommitted
#                        changes (the job runs `git archive HEAD`, never the tree)
#   -- <repo> <revision> run that revision of that clone instead of this checkout's HEAD
#   --learner-gpus N --server-nodes N (0: one shared 8-GPU node) --server-gpus-per-node N
#   --server-groups N --time T --exclude NODES
#   --dependency afterany:<jobid>  chains a resume segment behind a running one
#   Segments chain on their own until the run completes, fails or is stopped
#   (STOP_AUTO_RELAUNCH, or a TERM before the segment's last twelve minutes);
#   export ARC3_AUTO_RELAUNCH=0 to run a single segment.
#   --dry-run            submits the full allocation but only stages/prints commands
#   --test-only          asks Slurm to validate the submission without allocating anything;
#                        the run directory is not created, so a real submission can follow
# Environments (scripts/build_env.sbatch, docs/environments.md), passed to the job.
# Each one left unset is the newest artifact under the root built for the revision's
# locks and submodule pins, or, resuming, the one the run started with
# (scripts/resolve_envs.py); a set variable wins:
#   ARC3_FLASH_VENV_TARBALL         trainer-<extras>-<key>.tar.zst (the learner)
#   ARC3_HARNESS_VENV_TARBALL       harness-<key>.tar.zst (the servers; ARC-3 rollout workers)
#   ARC3_ARC2_WORKER_VENV_TARBALL   arc2-worker-vllm-fork-<key>.tar.zst (ARC-2 rollout workers),
#                                   required when the config's inference.environment is arc2
# Server concurrency: the job derives each engine's --max-num-seqs from the run's
# config (training.job_shape: ceil(parallel_rollouts / servers), capped by
# inference.max_concurrent_episodes, + 8), at every segment. An exported
# SERVER_MAX_NUM_SEQS overrides it for the whole chain; below that share the job refuses it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME=${USER:-$(id -un)}
LEARNER=4 NODES=1 GPUS=4 GROUPS_COUNT=1 TIME=06:00:00 EXCLUDE="" DEPENDENCY="" ALLOW_DIRTY=0 TEST_ONLY=0
RUN_DIR="${ARC3_RUN_DIR:-}" CONFIG="${ARC3_CONFIG:-}"
ENVS="${ARC_TRAINING_ENVS:-/data/personal/$USER_NAME/arc-training-envs}"
EXTRA=(--parsable)
while (( $# )); do
  case "$1" in
    --learner-gpus) LEARNER=${2:?}; shift 2 ;;
    --server-nodes) NODES=${2:?}; shift 2 ;;
    --server-gpus-per-node) GPUS=${2:?}; shift 2 ;;
    --server-groups) GROUPS_COUNT=${2:?}; shift 2 ;;
    --time) TIME=${2:?}; shift 2 ;;
    --exclude) EXCLUDE=${2:?}; shift 2 ;;
    --dependency) DEPENDENCY=${2:?}; shift 2 ;;
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    --config) CONFIG=${2:?}; shift 2 ;;
    --envs) ENVS=${2:?}; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --dry-run) export ARC3_DRY_RUN=1; shift ;;
    --test-only) EXTRA+=(--test-only); TEST_ONLY=1; shift ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$LEARNER" =~ ^[1-8]$ && "$GPUS" =~ ^[1-8]$ && "$NODES" =~ ^[0-9]+$ && "$GROUPS_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo 'Require 1..8 GPUs, nonnegative server nodes, and positive server groups' >&2; exit 2;
}
if [[ "$NODES" == 0 && ! ( "$LEARNER" == 4 && "$GPUS" == 4 && "$GROUPS_COUNT" == 1 ) ]]; then
  echo 'Single-allocation mode keeps the existing four learner + four server layout' >&2; exit 2
fi
[[ $# == 0 || $# == 2 ]] || { echo 'Expected -- <repo> <revision>' >&2; exit 2; }
[[ -n "$CONFIG" ]] || { echo 'Set --config <recipe> or ARC3_CONFIG; there is no default' >&2; exit 2; }
if (( $# == 0 )); then
  # The job archives a commit (git archive): default to this checkout's HEAD.
  REPO=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null) && [[ "$REPO/scripts" -ef "$HERE" ]] || {
    echo "$HERE is not at the root of a training repository: pass -- <repo> <revision> (see docs/environments.md#building)" >&2; exit 2;
  }
  # Submodule worktrees never run (the environments install the revision's pins);
  # a submodule checked out away from its pin is still reported.
  DIRTY=$(git -C "$REPO" status --porcelain --ignore-submodules=dirty)
  if [[ -n "$DIRTY" && "$ALLOW_DIRTY" != 1 ]]; then
    printf '%s\n' "$REPO has uncommitted changes, and the job runs its HEAD commit, not the working tree:" \
      "$DIRTY" "Commit them (a listed submodule: git submodule update), or pass --allow-dirty to run HEAD." >&2
    exit 2
  fi
  [[ -z "$DIRTY" ]] || echo "flash_rl_submit: --allow-dirty: running HEAD without the uncommitted changes" >&2
  set -- "$REPO" "$(git -C "$REPO" rev-parse --verify 'HEAD^{commit}')"
fi
REPO=$1 REVISION=$2
# The job gets the full commit, so it archives what the environments were resolved for
# even if a ref moves or an abbreviation becomes ambiguous. A clone this node cannot
# read is passed on as given (the job checks it).
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  REVISION=$(git -C "$REPO" rev-parse --verify --quiet "$2^{commit}") || {
    echo "$2 is not a commit in $REPO (fetch it, or pass another -- <repo> <revision>)" >&2; exit 2;
  }
fi
if [[ -z "$RUN_DIR" ]]; then
  STEM=${CONFIG##*/}
  RUN_DIR="/data/personal/$USER_NAME/arc-training-runs/${STEM%.json}-$(date +%Y%m%d-%H%M)/lora"
  [[ ! -e "$RUN_DIR" ]] || { echo "$RUN_DIR exists: pass --run-dir $RUN_DIR to resume it" >&2; exit 2; }
fi
[[ "$RUN_DIR" == /data/* ]] || { echo 'Set --run-dir /data/... or ARC3_RUN_DIR' >&2; exit 2; }

# The config the job will read: a resumed run's frozen copy, an absolute path, or
# a path inside the revision it archives.
read_config() {
  if [[ -f "$RUN_DIR/frozen_config.json" ]]; then cat "$RUN_DIR/frozen_config.json"
  elif [[ "$CONFIG" == /* ]]; then cat "$CONFIG"
  else git -C "$REPO" show "$REVISION:$CONFIG"
  fi
}
ENVIRONMENT=$(read_config 2>/dev/null | python3 -c 'import json, sys
print((json.load(sys.stdin).get("inference") or {}).get("environment", "arc3"))' 2>/dev/null) || ENVIRONMENT=""
if [[ -z "$ENVIRONMENT" ]]; then
  # Without the config only a fully specified submission can proceed; the job checks it.
  [[ -n "${ARC3_FLASH_VENV_TARBALL:-}" && -n "${ARC3_HARNESS_VENV_TARBALL:-}" ]] || {
    echo "Cannot read $CONFIG here (a relative path is read from $REVISION of $REPO): fix --config, or set" \
      "the ARC3_*_VENV_TARBALL variables to leave the config to the job" >&2
    exit 2
  }
  ENVIRONMENT=unknown
fi

NEED=()
[[ -n "${ARC3_FLASH_VENV_TARBALL:-}" ]] || NEED+=(trainer)
[[ -n "${ARC3_HARNESS_VENV_TARBALL:-}" ]] || NEED+=(harness)
[[ "$ENVIRONMENT" != arc2 || -n "${ARC3_ARC2_WORKER_VENV_TARBALL:-}" ]] || NEED+=(arc2-worker)
if (( ${#NEED[@]} )); then
  [[ -f "$HERE/resolve_envs.py" ]] || {
    echo "Set the ARC3_*_VENV_TARBALL variables for: ${NEED[*]} ($HERE has no resolve_envs.py)" >&2; exit 2;
  }
  STATUS=0
  # A resumed run keeps the environments it started with (the job refuses others).
  RESOLVED=$(python3 "$HERE/resolve_envs.py" --root "$ENVS" --repo "$REPO" --revision "$REVISION" \
    --run-dir "$RUN_DIR" "${NEED[@]}") || STATUS=$?
  while read -r recipe tarball; do
    case "$recipe" in
      trainer) export ARC3_FLASH_VENV_TARBALL="$tarball" ;;
      harness) export ARC3_HARNESS_VENV_TARBALL="$tarball" ;;
      arc2-worker) export ARC3_ARC2_WORKER_VENV_TARBALL="$tarball" ;;
    esac
  done <<< "$RESOLVED"
  if [[ "$STATUS" == 3 ]]; then
    # Pasted as is, it builds what this submission resolves, away from the nodes it excludes.
    BUILD=("$HERE/build_envs.sh" --revision "$REVISION" --out "$ENVS")
    [[ "$REPO" -ef "$HERE/.." ]] || BUILD+=(--repo "$REPO")
    [[ -z "$EXCLUDE" ]] || BUILD+=(-- "--exclude=$EXCLUDE")
    BUILD_LINE=$(printf '%q ' "${BUILD[@]}")
    {
      echo "Build the environments for $REVISION (CPU jobs; squeue -u $USER_NAME -n" \
        "build-env-trainer,build-env-harness,build-env-arc2-worker shows builds already running), then submit again:"
      echo "  ${BUILD_LINE% }"
    } >&2
    exit 2
  elif [[ "$STATUS" != 0 ]]; then
    echo "Cannot resolve the environments for: ${NEED[*]} (above); set ARC3_FLASH_VENV_TARBALL," \
      "ARC3_HARNESS_VENV_TARBALL or ARC3_ARC2_WORKER_VENV_TARBALL explicitly" >&2
    exit 2
  fi
fi

if [[ -n "${SERVER_MAX_NUM_SEQS:-}" ]]; then
  [[ "$SERVER_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]] || { echo 'SERVER_MAX_NUM_SEQS must be a positive integer' >&2; exit 2; }
  SEQS="SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS (from the environment, for every segment)"
else
  SEQS="SERVER_MAX_NUM_SEQS unset: each segment derives it from the run's config"
fi
{
  echo "flash_rl_submit: source $REPO at $REVISION"
  echo "flash_rl_submit: run directory $RUN_DIR"
  for var in ARC3_FLASH_VENV_TARBALL ARC3_HARNESS_VENV_TARBALL ARC3_ARC2_WORKER_VENV_TARBALL; do
    [[ -z "${!var:-}" ]] || echo "flash_rl_submit: $var=${!var}"
  done
  echo "flash_rl_submit: $SEQS"
} >&2
export ARC3_RUN_DIR="$RUN_DIR" ARC3_CONFIG="$CONFIG" ARC3_EXCLUDE="$EXCLUDE"
export ARC3_LEARNER_GPUS="$LEARNER" ARC3_SERVER_GPUS_PER_NODE="$GPUS"
# Keep the same optional prefix when mirroring both files into shared tools:
# H_flash_rl_submit.sh selects H_flash_rl.sbatch. A mirror outside a checkout
# takes its source from explicit -- <repo> <revision> arguments.
SCRIPT="${BASH_SOURCE[0]%_submit.sh}.sbatch"
[[ "$NODES" != 0 ]] || GROUPS_COUNT=0
args=(sbatch --export=ALL "${EXTRA[@]}")
# This cluster's sbatch ignores SBATCH_DEPENDENCY; the flag is the only way to chain segments.
[[ -z "$DEPENDENCY" ]] || args+=(--dependency="$DEPENDENCY")
for (( group=0; group<=GROUPS_COUNT; group++ )); do
  if (( group > 0 )); then args+=(:); fi
  args+=(--job-name=arc3-flash-rl --account=research --partition=dev --qos=b200-pool
    --cpus-per-gpu=12 --mem-per-gpu=96G --time="$TIME" --signal=B:TERM@600
    --output="$RUN_DIR/slurm-%j.out")
  if (( group == 0 )); then
    count="$LEARNER"
    [[ "$NODES" != 0 ]] || count=8
    args+=(--nodes=1 --ntasks=1 --gres="gpu:b200:$count")
  else
    args+=(--nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --gres="gpu:b200:$GPUS")
  fi
  [[ -z "$EXCLUDE" ]] || args+=(--exclude="$EXCLUDE")
done
# The job writes its Slurm output there, so it must exist before a real submission.
# Slurm checks a --test-only request without it, and creating it would make the
# default timestamped directory refuse a real submission in the same minute.
[[ "$TEST_ONLY" == 1 ]] || mkdir -p "$RUN_DIR"
args+=("$SCRIPT" "$REPO" "$REVISION")
printf '%q ' "${args[@]}"; printf '\n'
exec "${args[@]}"
