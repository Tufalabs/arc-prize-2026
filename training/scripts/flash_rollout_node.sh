#!/bin/bash
# One srun task per rollout node. The batch leader exports ARC3_RUN_DIR,
# ARC3_SERVER_JOB_ID (this component), ARC3_SERVER_SEGMENT_ID (leader job),
# ARC3_SERVER_NODE_OFFSET, ARC3_SERVER_GPUS_PER_NODE, SERVER_PORT (fleet base),
# ARC3_HARNESS_VENV_TARBALL, SERVER_API_KEY, the SERVER_* model shape and
# SERVER_MAX_NUM_SEQS (the job's, from the run's config).
# Source is the run's source.tar; nothing depends on another node's scratch.
set -euo pipefail
unset UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME
unset SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
export SLURM_JOB_ID="${ARC3_SERVER_JOB_ID:?}"
. /etc/kander/job-env.sh
umask 077
[[ "${SCRATCH_DIR:-}" == /scratch/* ]] || { echo 'Node scratch must be under /scratch' >&2; exit 1; }
mkdir -p "$SCRATCH_DIR"
WORK=$(mktemp -d "$SCRATCH_DIR/flash-rollout.XXXXXX")
PIDS=() CHILD=""
cleanup() {
  local status=$? pid deadline
  trap - TERM INT
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  if [[ -n "$CHILD" ]]; then kill -TERM -- "-$CHILD" 2>/dev/null || true; fi
  deadline=$((SECONDS + 30))
  for pid in "${PIDS[@]}" ${CHILD:+"$CHILD"}; do
    while kill -0 -- "-$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  cd /
  case "$WORK" in /scratch/*/flash-rollout.*) rm -rf -- "$WORK" ;; esac
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
run() {
  local status=0
  setsid "$@" & CHILD=$!
  wait "$CHILD" || status=$?
  CHILD=""
  return "$status"
}
RUN_DIR="${ARC3_RUN_DIR:?}"
NODE_INDEX=$(( ${ARC3_SERVER_NODE_OFFSET:-0} + ${SLURM_NODEID:?} ))
GPU_COUNT="${ARC3_SERVER_GPUS_PER_NODE:?}"
# SERVER_GPU=0..GPU_COUNT-1 below are this step's own GPUs only because kander
# confines a job step to its GRES devices (cgroup ConstrainDevices=yes) and
# numbers them from 0. Refuse a node that shows the step any other count.
if [[ "${ARC3_DRY_RUN:-0}" != 1 || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  VISIBLE_GPUS=$(nvidia-smi -L | grep -c '^GPU ') || VISIBLE_GPUS=0
  if [[ "$VISIBLE_GPUS" != "$GPU_COUNT" ]]; then
    echo "$(hostname) shows this step $VISIBLE_GPUS GPUs, not the $GPU_COUNT allocated: GPU indices" \
      "0..$(( GPU_COUNT - 1 )) are this allocation's only under GPU confinement (cgroup ConstrainDevices)" >&2
    exit 1
  fi
fi
BASE_PORT="${SERVER_PORT:?}"
: "${SERVER_API_KEY:?}" "${SERVER_MAX_NUM_SEQS:?}"
READY="$RUN_DIR/servers/${ARC3_SERVER_SEGMENT_ID:?}/node_${NODE_INDEX}.ready"
mkdir -p "$WORK/source" "$(dirname "$READY")"
run timeout -k 30s 5m tar -xf "$RUN_DIR/source.tar" -C "$WORK/source"
cd "$WORK/source"
export PYTHONPATH="$WORK/source" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT ARC3_SOURCE_WORKSPACE
run timeout -k 30s 15m bash -c 'set -euo pipefail; . "$1"; stage_venv_print "$2" HARNESS_PYTHON' \
  _ "$WORK/source/scripts/stage_venv.sh" "${ARC3_HARNESS_VENV_TARBALL:?}" > "$WORK/python"
HARNESS_PYTHON=$(cat "$WORK/python")
export PLATYPUS_VENV="${HARNESS_PYTHON%/bin/python}"
export ARC3_RL_CACHE_ROOT="/cache/users/$(id -u)/arc3-rl"
export ARC3_COMPILE_CACHE_ROOT="$ARC3_RL_CACHE_ROOT/compile"
export HF_HOME="$ARC3_RL_CACHE_ROOT/huggingface" TORCH_HOME="$ARC3_RL_CACHE_ROOT/torch"
export HF_HUB_CACHE="$HF_HOME/hub" XDG_CACHE_HOME="$ARC3_COMPILE_CACHE_ROOT/xdg-platypus"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export MAX_JOBS="$CMAKE_BUILD_PARALLEL_LEVEL"
export OMP_NUM_THREADS=$(( CMAKE_BUILD_PARALLEL_LEVEL / GPU_COUNT > 0 ? CMAKE_BUILD_PARALLEL_LEVEL / GPU_COUNT : 1 ))
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"
# Multiple engines read the checkpoint: copy once to node scratch. Preserve
# content/timestamps, not NFS ACLs (the S qualification established this recipe).
if [[ "${ARC3_DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "$WORK/checkpoint"
  run timeout -k 30s 30m cp -r --preserve=timestamps "$SERVER_MODEL/." "$WORK/checkpoint/"
  export SERVER_MODEL="$WORK/checkpoint"
fi
export SERVER_HOST=0.0.0.0 SERVER_TENSOR_PARALLEL=1
for (( gpu=0; gpu<GPU_COUNT; gpu++ )); do
  index=$((NODE_INDEX * GPU_COUNT + gpu))
  export SERVER_GPU="$gpu" SERVER_PORT=$((BASE_PORT + 100*index)) SERVER_LOG="$RUN_DIR/server_${index}.log"
  run timeout 15s "$HARNESS_PYTHON" -c 'import socket, sys
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = int(sys.argv[1])
    try:
        s.bind(("", port))
    except OSError:
        sys.exit(f"engine port {port} is in use on {socket.gethostname()}")' "$SERVER_PORT"
  # Print only this explicit allowlist. The API key is inherited, never argv/logs.
  printf '%q ' env "PLATYPUS_VENV=$PLATYPUS_VENV" "SERVER_HOST=$SERVER_HOST" \
    "SERVER_GPU=$SERVER_GPU" "SERVER_PORT=$SERVER_PORT" "SERVER_LOG=$SERVER_LOG" \
    "SERVER_MODEL=$SERVER_MODEL" "SERVER_WEIGHT_MODE=$SERVER_WEIGHT_MODE" \
    "SERVER_SERVED_NAME=$SERVER_SERVED_NAME" "SERVER_MAX_MODEL_LEN=$SERVER_MAX_MODEL_LEN" \
    "SERVER_ENVIRONMENT=$SERVER_ENVIRONMENT" "SERVER_MULTIMODAL_CONTEXT=$SERVER_MULTIMODAL_CONTEXT" \
    "SERVER_SCORE_CENTERING_TOP_K=$SERVER_SCORE_CENTERING_TOP_K" \
    "SERVER_SPECULATIVE_CONFIG=${SERVER_SPECULATIVE_CONFIG:-}" \
    "ARC3_RL_CACHE_ROOT=$ARC3_RL_CACHE_ROOT" "SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS" \
    setsid bash "$WORK/source/scripts/launch_platypus_server.sh"
  printf '\n'
  if [[ "${ARC3_DRY_RUN:-0}" != 1 ]]; then
    setsid bash "$WORK/source/scripts/launch_platypus_server.sh" & PIDS+=($!)
  fi
done
if [[ "${ARC3_DRY_RUN:-0}" != 1 ]]; then
  deadline=$((SECONDS + 6000))
  for (( gpu=0; gpu<GPU_COUNT; gpu++ )); do
    port=$((BASE_PORT + 100*(NODE_INDEX * GPU_COUNT + gpu)))
    until timeout -k 5s 15s "$HARNESS_PYTHON" -m training.server_endpoints "http://127.0.0.1:$port/models"; do
      kill -0 "${PIDS[$gpu]}" 2>/dev/null || { echo "Server $gpu died" >&2; exit 1; }
      (( SECONDS < deadline )) || { echo 'Server health deadline exceeded' >&2; exit 1; }
      sleep 10
    done
  done
fi
printf '%s\n' "$(hostname) $NODE_INDEX $GPU_COUNT" > "$READY.tmp"
mv "$READY.tmp" "$READY"
if [[ "${ARC3_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
# Any engine exit ends the task and tears down its siblings, even an exit 0.
wait -n "${PIDS[@]}" || exit $?
echo 'A rollout server exited unexpectedly' >&2
exit 1
