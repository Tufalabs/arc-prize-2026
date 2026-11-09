#!/bin/bash
#SBATCH --job-name=arc3-flash-rl
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --qos=b200-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm-%j.out
# Keep slurm-%j.out: external monitors key on this filename. Submit from /data
# or override --output with /data/.../slurm-%j.out.
# Required environment:
#   ARC3_RUN_DIR=/data/.../<run> (reuse for resume)
#   ARC3_CONFIG=<recipe> (no default; e.g. configs/flash_8gpu_arc2_grpo_tokmean.json)
#   ARC3_FLASH_VENV_TARBALL=/data/personal/$USER/arc-training-envs/trainer-flash-distillation-<key>.tar.zst
#   ARC3_HARNESS_VENV_TARBALL=/data/personal/$USER/arc-training-envs/harness-<key>.tar.zst
#   ARC3_ARC2_WORKER_VENV_TARBALL=/data/personal/$USER/arc-training-envs/arc2-worker-vllm-fork-<key>.tar.zst
#     (only when the config's inference.environment is arc2; ignored otherwise)
# Each requires an adjacent .json manifest; keys are the 16-hex filename suffixes. No environments
# are built by this job: scripts/build_env.sbatch builds them (docs/environments.md). The
# servers run from the harness; the rollout worker runs from the harness for ARC-3 and from
# the ARC-2 worker environment for ARC-2.
# Optional: ARC3_JOB_REPO_ROOT (checkout), ARC3_DRY_RUN=1, ARC3_AUTO_RELAUNCH=0 (segments chain by default),
# ARC3_SERVER_GPUS=0,1,2,3 (four TP=1 slots, disjoint from learner devices),
# SERVER_MAX_NUM_SEQS (each engine's --max-num-seqs). Unset, the job derives it from
# the run's config (training.job_shape: a server's share of a window plus 8); a value
# below that share is refused.
# Heterogeneous allocations: component 0 learns; components 1+ serve.
# Use scripts/flash_rl_submit.sh to compose the allocation.
# Alternatively: sbatch ... flash_rl.sbatch <fetch-only-repo> <revision>
# archives that revision to scratch; never checks out or changes the repository.
# CPU dry run: strip GPU SBATCH directives in a submission wrapper, request
# --qos=dev --cpus-per-task=8 --mem=16G --time=00:20:00, and export ARC3_DRY_RUN=1.
set -euo pipefail
unset UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME
# job-env preserves inherited values. Continuations must use this allocation's
# scratch/tmp paths, not directories from the segment that submitted them.
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
. /etc/kander/job-env.sh
umask 077

RUN_DIR="$(realpath -m "${ARC3_RUN_DIR:?set ARC3_RUN_DIR}")"
[[ "$RUN_DIR" == /data/* ]] || { echo 'ARC3_RUN_DIR must be under /data' >&2; exit 1; }
mkdir -p "$RUN_DIR"
WORK="$(mktemp -d "${SCRATCH_DIR:?}/flash-rl.XXXXXX")"
SERVER_PIDS=()
TRAINER_PID="" CHILD="" LOGGER_PID="" TRAINER_CACHE="" TRAINER_LAUNCHED=0
TRAINER_RC=0
# Seconds into the segment at which a TERM arrived; empty without one.
TERM_ELAPSED=""
STARTED_AT=$(date +%s)

# Every setup command is a waited process group, including timeout's children.
run() {
  local status=0
  setsid "$@" & CHILD=$!
  wait "$CHILD" || status=$?
  CHILD=""
  return "$status"
}
stop_group() {
  kill -KILL -- "-$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}
signal_trainer() {
  local launcher rank signalled=0
  # launch_frozen execs full_weight_launch -> torchrun -> learner ranks.
  # Sending TERM to torchrun itself starts its short worker-kill timeout;
  # signal the ranks directly so their iteration-boundary handlers get the
  # full checkpoint grace period. Before ranks exist, stop the launcher.
  for launcher in $(pgrep -P "$TRAINER_PID" || true); do
    for rank in $(pgrep -P "$launcher" || true); do
      kill -TERM "$rank" 2>/dev/null || true
      signalled=1
    done
  done
  if [[ "$signalled" == 0 ]]; then kill -TERM "$TRAINER_PID" 2>/dev/null || true; fi
}
on_signal() {
  # When, for maybe_relaunch: Slurm's end-of-job TERM and a scancel look alike.
  TERM_ELAPSED=$(( $(date +%s) - STARTED_AT ))
  # Signal the learner first. Keep rollout
  # servers alive while the trainer finishes its iteration and checkpoints.
  if [[ -n "$TRAINER_PID" ]]; then
    signal_trainer
  fi
  exit 143
}
# Segments of a run share the learner's compiled kernels through one archive
# in the run directory: Triton's binaries, the autotune picks FLA writes beside
# them (FLA_CACHE_RESULTS, on by default) and Inductor's graphs.
restore_kernel_cache() {
  local archive="$RUN_DIR/kernel_cache.tar.zst"
  [[ -f "$archive" ]] || return 0
  if run timeout -k 30s 10m tar -I zstd -xf "$archive" -C "$TRAINER_CACHE"; then
    echo "Restored the learner's kernel cache from $archive"
  else
    echo "Could not restore $archive; the learner starts with an empty kernel cache" >&2
    rm -rf -- "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  fi
}
save_kernel_cache() {
  local archive="$RUN_DIR/kernel_cache.tar.zst" budget=600
  # Stop before the time limit so cleanup still removes the node-local copy.
  if [[ -n "${LIMIT_SECONDS:-}" ]]; then
    budget=$((STARTED_AT + LIMIT_SECONDS - 60 - $(date +%s)))
    (( budget <= 600 )) || budget=600
  fi
  if (( budget < 30 )); then
    echo "No time left to save the kernel cache; $archive is unchanged" >&2
    return 0
  fi
  if run timeout -k 30s "${budget}s" tar -I zstd -cf "$archive.tmp" -C "$TRAINER_CACHE" triton inductor &&
    mv -f -- "$archive.tmp" "$archive"; then
    echo "Saved the learner's kernel cache to $archive"
  else
    rm -f -- "$archive.tmp"
    echo "Could not save the kernel cache; $archive is unchanged" >&2
  fi
}
cleanup() {
  local status=$? deadline pid
  trap - TERM INT
  if [[ -n "$TRAINER_PID" ]]; then
    signal_trainer
    deadline=$((SECONDS + 540))
    while kill -0 "$TRAINER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    stop_group "$TRAINER_PID"
  fi
  for pid in "${SERVER_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  deadline=$((SECONDS + 30))
  for pid in "${SERVER_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    stop_group "$pid"
  done
  if [[ -n "$CHILD" ]]; then
    kill -TERM -- "-$CHILD" 2>/dev/null || true
    sleep 1
    stop_group "$CHILD"
  fi
  if [[ -n "$LOGGER_PID" ]]; then
    kill -TERM "$LOGGER_PID" 2>/dev/null || true
    wait "$LOGGER_PID" 2>/dev/null || true
  fi
  maybe_relaunch || true
  # After the submission: the next segment starts only when this job has ended.
  if [[ "${TRAINER_LAUNCHED:-0}" == 1 ]]; then save_kernel_cache || true; fi
  cd /
  case "${TRAINER_CACHE:-}" in
    "/cache/users/$(id -u)/arc3-rl/compile/trainer-run-${RUN_CACHE_KEY:-none}") rm -rf -- "$TRAINER_CACHE" ;;
  esac
  case "$WORK" in /scratch/*/flash-rl.*) rm -rf -- "$WORK" ;; esac
  exit "$status"
}
# Defined early so cleanup also works after a failed setup step.
maybe_relaunch() { :; }
trap cleanup EXIT
trap on_signal TERM INT

SELF_ROOT="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
[[ -f "$SELF_ROOT/scripts/flash_rl.sbatch" ]] || SELF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${ARC3_JOB_REPO_ROOT:-$SELF_ROOT}"
SOURCE_REPO="$(realpath "${1:-$REPO_ROOT}")"
REVISION="${2:-}"
if [[ -n "$REVISION" ]]; then
  mkdir -p "$WORK/source"
  run timeout -k 30s 5m bash -c 'set -euo pipefail; git -C "$1" archive "$2" | tar -x -C "$3"' \
    _ "$SOURCE_REPO" "$REVISION" "$WORK/source"
  REPO_ROOT="$WORK/source"
fi
cd "$REPO_ROOT"
REPO_ROOT="$PWD"
[[ -f scripts/flash_rl.sbatch ]] || { echo 'Not an ARC3 checkout' >&2; exit 1; }
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT PYTHONPATH ARC3_INFERENCE_BASE_URL
export PYTHONPATH="$REPO_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ARC3_RUN_DIR="$RUN_DIR" ARC3_SOURCE_WORKSPACE="$WORK/frozen-source"
export ARC3_RL_CACHE_ROOT="/cache/users/$(id -u)/arc3-rl"
export ARC3_COMPILE_CACHE_ROOT="$ARC3_RL_CACHE_ROOT/compile"
export HF_HOME="$ARC3_RL_CACHE_ROOT/huggingface" TORCH_HOME="$ARC3_RL_CACHE_ROOT/torch"
export HF_HUB_CACHE="$HF_HOME/hub"
# FLA persists autotune results: isolate trainer picks from other workloads.
# One directory per run, not per job: Triton's cache index records absolute
# paths, so a segment restores the run's archive at the path that wrote it.
RUN_CACHE_KEY="$(printf '%s' "$RUN_DIR" | sha256sum | cut -c1-16)"
TRAINER_CACHE="$ARC3_COMPILE_CACHE_ROOT/trainer-run-$RUN_CACHE_KEY"
export TRITON_CACHE_DIR="$TRAINER_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$TRAINER_CACHE/inductor"
export XDG_CACHE_HOME="$ARC3_COMPILE_CACHE_ROOT/xdg-trainer"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export MAX_JOBS="$CMAKE_BUILD_PARALLEL_LEVEL"
export OMP_NUM_THREADS="$(( CMAKE_BUILD_PARALLEL_LEVEL / 8 > 0 ? CMAKE_BUILD_PARALLEL_LEVEL / 8 : 1 ))"
mkdir -p "$ARC3_RL_CACHE_ROOT" "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

if [[ "${ARC3_AUTO_RELAUNCH:-1}" == 1 && -f "$RUN_DIR/STOP_AUTO_RELAUNCH" ]]; then
  echo 'STOP_AUTO_RELAUNCH present; ending the chain.'
  exit 0
fi
limit_seconds() {
  printf '%s' "$1" | awk -F '[-:]' 'NF==4 {print (($1*24+$2)*60+$3)*60+$4} NF==3 {print ($1*60+$2)*60+$3}'
}
# Querying a heterogeneous leader returns every component. Select its record.
JOB_INFO="$(timeout 15s scontrol show job -o "${SLURM_JOB_ID:?}" 2>/dev/null | awk 'NR == 1 {print}' || true)"
TIME_LIMIT="$(printf '%s' "$JOB_INFO" | sed -n 's/.* TimeLimit=\([^ ]*\).*/\1/p')"
LIMIT_SECONDS="$(limit_seconds "$TIME_LIMIT")"
if [[ -n "$LIMIT_SECONDS" && -z "${ARC3_SEGMENT_DEADLINE_EPOCH:-}" ]]; then
  export ARC3_SEGMENT_DEADLINE_EPOCH=$((STARTED_AT + LIMIT_SECONDS - 600))
fi
# Slurm's end-of-job TERM (--signal=B:TERM@600) comes ten minutes before the
# limit, or up to a minute earlier; a TERM inside this window is taken as that one.
RELAUNCH_WINDOW=720
# Chain the next segment after a learner that yielded at a checkpoint (exit 75)
# or after the end-of-job TERM, judged by when the TERM arrived: the trap
# interrupts `wait` and on_signal exits before the learner's status is read, so
# TRAINER_RC is never 75 on that path, and a learner waiting on a slow
# collection stops within seconds, long before the limit.
# Every other ending prints why it does not chain.
maybe_relaunch() {
  local info state next_limit why="" reason=""
  if [[ "${ARC3_AUTO_RELAUNCH:-1}" != 1 ]]; then why='ARC3_AUTO_RELAUNCH is not 1'
  elif [[ "${ARC3_DRY_RUN:-0}" == 1 ]]; then why='a dry run'
  elif [[ -f "$RUN_DIR/STOP_AUTO_RELAUNCH" ]]; then why="$RUN_DIR/STOP_AUTO_RELAUNCH exists"
  else
    info="$(timeout 15s scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | awk 'NR == 1 {print}' || true)"
    state="$(printf '%s' "$info" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')"
    if [[ "$state" == CANCELLED* ]]; then why="the job was cancelled (JobState=$state)"
    elif [[ "$TRAINER_RC" == 75 ]]; then reason='the learner yielded at a checkpoint (exit 75)'
    elif [[ -n "$TERM_ELAPSED" ]]; then
      case "$state" in
        TIMEOUT|PREEMPTED|NODE_FAIL) reason="TERM with JobState=$state" ;;
        *)
          if [[ -z "$LIMIT_SECONDS" ]]; then
            why="TERM ${TERM_ELAPSED}s into the segment, whose time limit is unknown"
          elif (( TERM_ELAPSED >= LIMIT_SECONDS - RELAUNCH_WINDOW )); then
            reason="TERM ${TERM_ELAPSED}s into a ${LIMIT_SECONDS}s limit, the end-of-job signal"
          else
            why="TERM ${TERM_ELAPSED}s into a ${LIMIT_SECONDS}s limit, before its last ${RELAUNCH_WINDOW}s: a stop, not the end-of-job signal"
          fi
          ;;
      esac
    elif [[ "${TRAINER_LAUNCHED:-0}" != 1 ]]; then why='the job ended before the learner started'
    else why="the learner exited $TRAINER_RC without yielding at a checkpoint (75)"
    fi
  fi
  if [[ -n "$why" ]]; then
    echo "Not relaunching: $why."
    return 0
  fi
  echo "Relaunching: $reason."
  next_limit="$(cat "$RUN_DIR/NEXT_TIME_LIMIT" 2>/dev/null || true)"
  next_limit="${next_limit:-${TIME_LIMIT:-06:00:00}}"
  if [[ "${SLURM_HET_SIZE:-1}" -gt 1 ]]; then
    relaunch_het "$next_limit"
    return
  fi
  # Submit before deleting scratch: sbatch spools the script immediately.
  # The fetch-only source/revision arguments also survive node changes, and the
  # submit helper's node exclusions (ARC3_EXCLUDE) hold for the whole chain.
  local -a args=(--parsable --dependency="afterany:$SLURM_JOB_ID"
    --qos="${ARC3_QOS:-b200-pool}" --time="$next_limit"
    --chdir="$RUN_DIR" --output="$RUN_DIR/slurm-%j.out" --export=ALL)
  if [[ -n "${ARC3_EXCLUDE:-}" ]]; then args+=(--exclude="$ARC3_EXCLUDE"); fi
  env -u TMPDIR -u CUDA_VISIBLE_DEVICES -u ARC3_SEGMENT_DEADLINE_EPOCH \
    -u ARC3_FORCE_SNAPSHOT -u ARC3_JOB_REPO_ROOT \
    timeout 30s sbatch "${args[@]}" "$REPO_ROOT/scripts/flash_rl.sbatch" "$SOURCE_REPO" "${REVISION:-HEAD}"
}

# Stage each immutable tarball under an outer deadline, including lock wait.
stage() {
  local tarball=$1 base key
  base="${tarball##*/}"; base="${base%.tar.zst}"; key="${base##*-}"
  [[ "$key" =~ ^[0-9a-f]{16}$ && "$tarball" == *.tar.zst ]] || {
    echo "Expected <name>-<16-hex-key>.tar.zst: $tarball" >&2; return 1;
  }
  run timeout -k 30s 15m bash -c 'set -euo pipefail; . "$1"; stage_venv_print "$2" STAGED_PYTHON' \
    _ "$REPO_ROOT/scripts/stage_venv.sh" "$tarball"
}
stage "${ARC3_FLASH_VENV_TARBALL:?set ARC3_FLASH_VENV_TARBALL}" > "$WORK/trainer-python"
stage "${ARC3_HARNESS_VENV_TARBALL:?set ARC3_HARNESS_VENV_TARBALL}" > "$WORK/harness-python"
export ARC3_FLASH_PYTHON="$(cat "$WORK/trainer-python")"
HARNESS_PYTHON="$(cat "$WORK/harness-python")"
export PLATYPUS_VENV="${HARNESS_PYTHON%/bin/python}"

# CPU launch preparation: effective config, free ports, atomic resume rewrite.
SRC_CONFIG="${ARC3_CONFIG:?set ARC3_CONFIG}"
[[ ! -f "$RUN_DIR/frozen_config.json" ]] || SRC_CONFIG="$RUN_DIR/frozen_config.json"
LAUNCH_CONFIG="$RUN_DIR/launch_config.json"
# The rollout worker's environment (inference.platypus_venv). ARC-3 runs it
# from the harness; ARC-2's packages share names with the harness's, so ARC-2
# brings its own environment. The servers always run from the harness.
WORKER_VENV="$PLATYPUS_VENV"
TRAINING_ENVIRONMENT="$("$ARC3_FLASH_PYTHON" -c 'import json, sys
print(json.load(open(sys.argv[1]))["inference"].get("environment", "arc3"))' "$SRC_CONFIG")"
if [[ "$TRAINING_ENVIRONMENT" == arc2 ]]; then
  stage "${ARC3_ARC2_WORKER_VENV_TARBALL:?ARC-2 configs need ARC3_ARC2_WORKER_VENV_TARBALL (build_env.sbatch arc2-worker)}" \
    > "$WORK/worker-python"
  WORKER_PYTHON="$(cat "$WORK/worker-python")"
  WORKER_VENV="${WORKER_PYTHON%/bin/python}"
  # The worker checks its tokenizer and image processing against the engine's
  # versions; refuse a mismatched pair here rather than at the first image.
  STACK=(transformers tokenizers torch torchvision pillow numpy vllm)
  run timeout 120s "$PLATYPUS_VENV/bin/python" -I -B "$REPO_ROOT/scripts/env_checks.py" versions "${STACK[@]}" \
    > "$WORK/harness-stack"
  run timeout 120s "$WORKER_PYTHON" -I -B "$REPO_ROOT/scripts/env_checks.py" versions "${STACK[@]}" \
    > "$WORK/worker-stack"
  if ! cmp -s "$WORK/harness-stack" "$WORK/worker-stack"; then
    echo 'The ARC-2 worker and harness environments differ in their serving stack (harness, worker):' >&2
    paste -d ' ' "$WORK/harness-stack" "$WORK/worker-stack" >&2
    exit 1
  fi
  # The trainer reads the pinned puzzle sets from the source tree's ARC-2
  # submodule (harness_support/arc2/dataset.py), which an archived revision
  # lacks: take them from the worker's installed taaf, built from that pin.
  DATA="arc2-platypus/packages/taaf/src/taaf/data"
  if ! [[ -f "$REPO_ROOT/$DATA/arc2-train.npz" && -f "$REPO_ROOT/$DATA/arc2-eval.npz" ]]; then
    [[ "$REPO_ROOT" == "$WORK/source" ]] || {
      echo "$REPO_ROOT has no $DATA; initialize the arc2-platypus submodule" >&2; exit 1;
    }
    run timeout 60s "$WORKER_PYTHON" -I -c 'import importlib.util, pathlib, shutil, sys
data = pathlib.Path(importlib.util.find_spec("taaf").submodule_search_locations[0]) / "data"
target = pathlib.Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
for name in ("arc2-train.npz", "arc2-eval.npz"):
    shutil.copyfile(data / name, target / name)' "$REPO_ROOT/$DATA"
  fi
fi
port_free() {
  "$ARC3_FLASH_PYTHON" -c 'import socket, sys
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", int(sys.argv[1])))' "$1" 2>/dev/null
}
if [[ "${SLURM_HET_SIZE:-1}" -gt 1 ]]; then
  source "$REPO_ROOT/scripts/flash_rl_het.sh"
  prepare_het
else
run timeout 60s "$ARC3_FLASH_PYTHON" -m training.job_shape "$SRC_CONFIG" --flash-job > "$WORK/shape"
source "$WORK/shape"
read -r -a SERVER_GPUS <<< "$SERVER_GPU_LIST"
NUM_SERVERS=${#SERVER_GPUS[@]}
[[ -z "${SERVER_URL_LIST:-}" ]] || { echo 'Separate servers require a heterogeneous allocation' >&2; exit 1; }
SERVER_PORT=$((50000 + SLURM_JOB_ID % 4000))
PORTS_OK=0
for (( attempt=0; attempt<40; attempt++ )); do
  PORTS_OK=1
  for (( i=0; i<NUM_SERVERS; i++ )); do
    if ! port_free $((SERVER_PORT + 100*i)) || ! port_free $((SERVER_PORT + 100*i + 5000)); then
      PORTS_OK=0; break
    fi
  done
  [[ "$PORTS_OK" != 1 ]] || break
  SERVER_PORT=$((SERVER_PORT + 101))
done
[[ "$PORTS_OK" == 1 ]] || { echo 'No free server port block' >&2; exit 1; }
run timeout 60s "$ARC3_FLASH_PYTHON" -m training.job_shape "$SRC_CONFIG" --flash-job \
  --launch-config "$LAUNCH_CONFIG" --port "$SERVER_PORT" --venv "$WORKER_VENV" > "$WORK/shape"
fi
if [[ -f "$RUN_DIR/frozen_config.json" ]]; then
  run timeout 60s "$ARC3_FLASH_PYTHON" -m training.job_shape "$LAUNCH_CONFIG" --flash-job \
    --launch-config "$RUN_DIR/frozen_config.json" --port "$SERVER_PORT" --venv "$WORKER_VENV" > "$WORK/shape"
fi

# Prepare the single source snapshot and fingerprint before starting servers.
run timeout -k 30s 5m "$ARC3_FLASH_PYTHON" -m training.launch_frozen \
  --config "$LAUNCH_CONFIG" --run-dir "$RUN_DIR" --prepare-only > "$WORK/frozen-root"
export ARC3_FROZEN_REPO_ROOT="$(cat "$WORK/frozen-root")"
export ARC3_ORIGINAL_REPO_ROOT="$SOURCE_REPO"
export ARC3_FROZEN_CONFIG_PATH="$RUN_DIR/frozen_config.json"
export ARC3_RUNTIME_FINGERPRINT_PATH="$RUN_DIR/runtime_fingerprint.json"
export PYTHONPATH="$ARC3_FROZEN_REPO_ROOT"
unset ARC3_FORCE_SNAPSHOT
cd "$ARC3_FROZEN_REPO_ROOT"
run timeout 60s "$ARC3_FLASH_PYTHON" -m training.job_shape "$ARC3_FROZEN_CONFIG_PATH" --flash-job > "$WORK/shape"
source "$WORK/shape"
read -r -a SERVER_GPUS <<< "$SERVER_GPU_LIST"
read -r -a SERVER_CONTEXT_WINDOWS <<< "$SERVER_CONTEXT_LIST"
export SERVER_MODEL SERVER_REVISION SERVER_SERVED_NAME
export SERVER_WEIGHT_MODE SERVER_SCORE_CENTERING_TOP_K SERVER_ENVIRONMENT SERVER_MULTIMODAL_CONTEXT
export SERVER_SPECULATIVE_CONFIG
export SERVER_TENSOR_PARALLEL=1 CUDA_VISIBLE_DEVICES="$DEFAULT_VISIBLE_GPUS"
# These indices (the learner's device_ids, the shared node's server GPUs) are
# this allocation's GPUs only because kander confines a job to its GRES devices
# (cgroup ConstrainDevices=yes) and numbers them from 0. Refuse a node that
# shows the job any other count than the GPUs it binds.
if [[ "${ARC3_DRY_RUN:-0}" != 1 || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  IFS=, read -r -a BOUND_GPUS <<< "$DEFAULT_VISIBLE_GPUS"
  VISIBLE_GPUS=$(nvidia-smi -L | grep -c '^GPU ') || VISIBLE_GPUS=0
  if [[ "$VISIBLE_GPUS" != "${#BOUND_GPUS[@]}" ]]; then
    echo "$(hostname) shows this job $VISIBLE_GPUS GPUs, not the ${#BOUND_GPUS[@]} it binds as" \
      "$DEFAULT_VISIBLE_GPUS: those are this allocation's only under GPU confinement (cgroup ConstrainDevices)" >&2
    exit 1
  fi
fi
export FLA_TILELANG=0 FLA_CONV_BACKEND=triton
print_command() { printf '%q ' "$@"; printf '\n'; }
echo "Flash learner GPUs: $TRAINER_GPUS; base port: $SERVER_PORT; run: $RUN_DIR; rollout worker: $WORKER_VENV"
# The frozen config's shape sets it: each server's share of a window plus 8, or an
# explicit SERVER_MAX_NUM_SEQS of at least that share. A derived value is not
# exported, so a relaunched segment derives it again from its frozen config.
echo "Each engine's --max-num-seqs: $SERVER_MAX_NUM_SEQS"
if [[ "${SLURM_HET_SIZE:-1}" -gt 1 ]]; then
  read -r -a SERVER_URLS <<< "$SERVER_URL_LIST"
  launch_het_servers
else
# Four engines read the checkpoint: copy it to node scratch once (sequential
# NFS read) instead of four concurrent NFS loads, which took 13 minutes each.
if [[ "${ARC3_DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "$WORK/checkpoint"
  run timeout -k 30s 30m cp -r --preserve=timestamps "$SERVER_MODEL/." "$WORK/checkpoint/"
  SERVER_MODEL="$WORK/checkpoint"
fi
for (( i=0; i<NUM_SERVERS; i++ )); do
  SERVER_ENV=(
    "PLATYPUS_VENV=$PLATYPUS_VENV" "SERVER_WEIGHT_MODE=$SERVER_WEIGHT_MODE"
    "SERVER_MODEL=$SERVER_MODEL" "SERVER_REVISION=$SERVER_REVISION" "SERVER_SERVED_NAME=$SERVER_SERVED_NAME"
    "SERVER_MAX_MODEL_LEN=${SERVER_CONTEXT_WINDOWS[$i]}" "SERVER_GPU=${SERVER_GPUS[$i]}"
    "SERVER_PORT=$((SERVER_PORT + 100*i))" "SERVER_TENSOR_PARALLEL=1"
    "SERVER_ENVIRONMENT=$SERVER_ENVIRONMENT" "SERVER_MULTIMODAL_CONTEXT=$SERVER_MULTIMODAL_CONTEXT"
    "SERVER_SCORE_CENTERING_TOP_K=$SERVER_SCORE_CENTERING_TOP_K"
    "SERVER_SPECULATIVE_CONFIG=$SERVER_SPECULATIVE_CONFIG"
    "ARC3_RL_CACHE_ROOT=$ARC3_RL_CACHE_ROOT" "SERVER_LOG=$RUN_DIR/server_${i}.log"
    "OMP_NUM_THREADS=$OMP_NUM_THREADS" "PYTHONUNBUFFERED=1"
    "SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS"
  )
  SERVER_COMMAND=(env "${SERVER_ENV[@]}" setsid bash "$ARC3_FROZEN_REPO_ROOT/scripts/launch_platypus_server.sh")
  print_command "${SERVER_COMMAND[@]}"
  if [[ "${ARC3_DRY_RUN:-0}" != 1 ]]; then
    "${SERVER_COMMAND[@]}" &
    SERVER_PIDS+=($!)
  fi
done
fi
TRAINER_COMMAND=(env "ARC3_FLASH_PYTHON=$ARC3_FLASH_PYTHON" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "OMP_NUM_THREADS=$OMP_NUM_THREADS" "PYTHONUNBUFFERED=1"
  "$ARC3_FLASH_PYTHON" -m training.launch_frozen --config "$LAUNCH_CONFIG" --run-dir "$RUN_DIR")
print_command setsid "${TRAINER_COMMAND[@]}"
if [[ "${ARC3_DRY_RUN:-0}" == 1 ]]; then
  echo 'ARC3_DRY_RUN=1: staged and prepared; no servers or trainer launched.'
  exit 0
fi
# Unpack the run's kernel cache while the servers load.
restore_kernel_cache

# Health requests have a 2s HTTP timeout plus an outer process deadline.
HEALTH_DEADLINE=$((SECONDS + 6000))
for (( i=0; i<NUM_SERVERS; i++ )); do
  until timeout -k 5s 15s "${PLATYPUS_VENV}/bin/python" -m training.server_endpoints \
    "${SERVER_URLS[$i]:-http://127.0.0.1:$((SERVER_PORT + 100*i))}/models" > /dev/null 2>&1; do
    for pid in "${SERVER_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -40 "$RUN_DIR/server_${i}.log" >&2 || true
      echo "Server $i died during startup" >&2; exit 1
    fi
    done
    (( SECONDS < HEALTH_DEADLINE )) || { echo "Server $i health deadline exceeded" >&2; exit 1; }
    sleep 10
  done
done

# A FIFO preserves the trainer's PID/exit status and lets cleanup reap tee.
mkfifo "$WORK/trainer-output"
tee -a "$RUN_DIR/trainer.log" < "$WORK/trainer-output" & LOGGER_PID=$!
TRAINER_LAUNCHED=1
setsid "${TRAINER_COMMAND[@]}" > "$WORK/trainer-output" 2>&1 & TRAINER_PID=$!
wait "$TRAINER_PID" || TRAINER_RC=$?
stop_group "$TRAINER_PID"
TRAINER_PID=""
wait "$LOGGER_PID" || true
LOGGER_PID=""
if [[ "$TRAINER_RC" == 75 ]]; then echo 'Trainer yielded at a safe checkpoint boundary (75).'; fi
exit "$TRAINER_RC"
