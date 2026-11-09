#!/bin/bash
#SBATCH --job-name=arc3-eval-lora-1gpu
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
# --output must stay slurm-%j.out (not %x): external monitors key on the
# slurm-<jobid>.out filename. No #SBATCH --qos: pass `sbatch --qos=...` when
# the allocation needs one.
#
# Paired 2-GPU evaluation, NO training and NO auto-relaunch: GPU 0 serves the
# plain base model, GPU 1 serves base + the trained LoRA (hot-loaded by the
# driver via vLLM's runtime-LoRA endpoint, same contract as training). The
# driver (scripts/eval_base_vs_lora.py) runs 25 games x 10 fixed seeds per
# arm with continuous backfill, then exits; analysis runs offline via
# scripts/eval_base_vs_lora_analysis.py <eval_dir>.
#
# Knobs (all optional):
#   ARC3_EVAL_DIR          output dir (default under /data/personal/$USER/eval_runs)
#   ARC3_EVAL_ADAPTER_DIR  trainer-format adapter checkpoint to evaluate
#   ARC3_EVAL_CONFIG       config for game list + agent knobs
#   ARC3_EVAL_DRIVER_ARGS  extra driver args (e.g. "--passes 2" for a smoke)
set -euo pipefail

# Default repo root: the checkout this script lives in. Under sbatch the
# script executes from a spooled copy, so fall back to the submit directory;
# ARC3_JOB_REPO_ROOT overrides both.
__arc3_self_root="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
if [ ! -f "${__arc3_self_root}/scripts/eval_base_vs_lora.sbatch" ]; then
  __arc3_self_root="${SLURM_SUBMIT_DIR:-${PWD}}"
fi
REPO_ROOT="${ARC3_JOB_REPO_ROOT:-${__arc3_self_root}}"
if [ ! -f "${REPO_ROOT}/scripts/eval_base_vs_lora.sbatch" ]; then
  echo "FATAL: ${REPO_ROOT} is not an ARC3 checkout; export ARC3_JOB_REPO_ROOT." >&2
  exit 1
fi
cd "$REPO_ROOT"

# Kander's job env is required (TMPDIR on job-local disk, cluster paths): a
# node without it must fail the job, not fall back to host defaults.
. /etc/kander/job-env.sh

# 100 concurrent episodes hold subprocess pipes, sockets, and tokenizer file
# handles; the default nofile limit exhausts mid-run and crashes every
# episode with EMFILE. Raise soft to hard before anything opens.
ulimit -n "$(ulimit -Hn)" || true
echo "nofile limit: $(ulimit -n)"

EVAL_DIR="${ARC3_EVAL_DIR:-/data/personal/$USER/eval_runs/$(date +%Y%m%d_%H%M%S)_base_vs_g25hidden87_80k}"
ADAPTER_DIR="${ARC3_EVAL_ADAPTER_DIR:-/data/personal/$USER/training_runs/20260829_073025_platypus_games25_hidden/checkpoints/adapter_iter_000087}"
EVAL_CONFIG="${ARC3_EVAL_CONFIG:-configs/ppo_2gpu_arc3_games25_platypus.json}"
mkdir -p "$EVAL_DIR"
# A pre-seeded runtime export (e.g. carried over from a prior eval of a
# checkpoint the training run has since garbage-collected) supersedes the
# trainer-format source: the driver reuses it and never reads ADAPTER_DIR.
if [ -f "${EVAL_DIR}/runtime_vllm_adapter/adapter_model.safetensors" ]; then
  echo "using pre-seeded runtime adapter at ${EVAL_DIR}/runtime_vllm_adapter"
elif [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
  echo "FATAL: ${ADAPTER_DIR} is not a trainer-format adapter checkpoint." >&2
  exit 1
fi

# --- node-local caches + venv (identical recipe to platypus_debug_smoke.sbatch
# so this job REUSES the training venv on any node that already built it; the
# venv key covers everything that determines its content) -------------------
umask 002
ARC3_RL_CACHE_ROOT="${ARC3_RL_CACHE_ROOT:-/cache/shared/research/arc-rl}"
if ! { [ -d "$(dirname "${ARC3_RL_CACHE_ROOT}")" ] && mkdir -p "${ARC3_RL_CACHE_ROOT}" 2>/dev/null && [ -w "${ARC3_RL_CACHE_ROOT}" ]; }; then
  echo "FATAL: node-local cache root ${ARC3_RL_CACHE_ROOT} is missing or not writable on $(hostname)." >&2
  echo "       /cache/shared/research must be provisioned (group research, setgid) on every RL node; ask the admins, or submit with --exclude=$(hostname)." >&2
  exit 1
fi
echo "rl cache root: ${ARC3_RL_CACHE_ROOT} (shared)"
export ARC3_RL_CACHE_ROOT
export ARC3_COMPILE_CACHE_ROOT="${ARC3_COMPILE_CACHE_ROOT:-${ARC3_RL_CACHE_ROOT}/compile}"
export TRITON_CACHE_DIR="${ARC3_COMPILE_CACHE_ROOT}/triton-trainer"
export TORCHINDUCTOR_CACHE_DIR="${ARC3_COMPILE_CACHE_ROOT}/inductor-trainer"
export XDG_CACHE_HOME="${ARC3_COMPILE_CACHE_ROOT}/xdg-trainer"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}"
export HF_HOME="${ARC3_RL_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"

bash "${REPO_ROOT}/scripts/sync_platypus.sh"
VENV_ROOT="${VENV_ROOT:-${ARC3_RL_CACHE_ROOT}/venvs}"
export UV_CACHE_DIR="/cache/users/$(id -u)/uv"
mkdir -p "${UV_CACHE_DIR}"
PLATYPUS_SHA="$(git -C platypus rev-parse HEAD)"
export ARC3_FAST_KERNELS="${ARC3_FAST_KERNELS:-1}"
FAST_KERNEL_RECIPE="flash-linear-attention==0.5.1 fla-core==0.5.1 causal-conv1d==1.6.2.post1"
LOCK_HASH="$({ sha256sum uv.lock; printf 'platypus=%s\n' "${PLATYPUS_SHA}"; printf 'recipe=v6\n'; printf 'fast_kernels=%s %s\n' "${ARC3_FAST_KERNELS}" "${FAST_KERNEL_RECIPE}"; uv --version; } | sha256sum | cut -c1-16)"
TRAIN_VENV="${VENV_ROOT}/${LOCK_HASH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export MAX_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL}"
__arc3_export_cuda_env() {
  local delta key value
  delta="$("${TRAIN_VENV}/bin/python" - <<'PY'
import os
before = dict(os.environ)
from ullm.backends import config_local
config_local.set_env()
for key, value in os.environ.items():
    if before.get(key) != value:
        print(f"{key}={value}")
PY
)"
  while IFS="=" read -r key value; do
    if [ -n "${key}" ]; then export "${key}=${value}"; fi
  done <<< "${delta}"
}
mkdir -p "${VENV_ROOT}"
exec 9>"${VENV_ROOT}/${LOCK_HASH}.lock"
flock 9
if [ ! -f "${TRAIN_VENV}/.arc3-ready" ]; then
  echo "building venv ${TRAIN_VENV} from uv.lock"
  build_started=$(date +%s)
  BUILD_SRC="$(mktemp -d "${TMPDIR:?TMPDIR must come from /etc/kander/job-env.sh}/arc3-venv-build-${SLURM_JOB_ID}.XXXX")"
  # The whole monorepo: uv's path sources are sibling directories of training/.
  git -C "$(git rev-parse --show-toplevel)" archive HEAD | tar -x -C "${BUILD_SRC}"
  (cd "${BUILD_SRC}/training" && env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="${TRAIN_VENV}" uv sync --locked --extra harness --no-editable)
  rm -rf "${BUILD_SRC}"
  if [ "${ARC3_FAST_KERNELS}" != "0" ]; then
    __arc3_export_cuda_env
    env -u VIRTUAL_ENV uv pip install --python "${TRAIN_VENV}/bin/python" \
      flash-linear-attention==0.5.1 fla-core==0.5.1
    TORCH_VERSION="$(grep -A1 '^name = "torch"$' uv.lock | grep -oE 'version = "[0-9.]+"' | cut -d'"' -f2)"
    CONV1D_WHEEL="${ARC3_CAUSAL_CONV1D_WHEEL:-${ARC3_RL_CACHE_ROOT}/wheels/py312-torch${TORCH_VERSION}-cu130-b200/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl}"
    if [ -f "${CONV1D_WHEEL}" ]; then
      echo "installing prebuilt causal-conv1d for torch ${TORCH_VERSION}: ${CONV1D_WHEEL}"
      env -u VIRTUAL_ENV uv pip install --python "${TRAIN_VENV}/bin/python" "${CONV1D_WHEEL}"
    else
      echo "building causal-conv1d from source for torch ${TORCH_VERSION} (one-time per venv, cache bypassed)"
      env -u VIRTUAL_ENV uv pip install --python "${TRAIN_VENV}/bin/python" --no-cache \
        --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.6.2.post1
    fi
    "${TRAIN_VENV}/bin/python" -c "import torch, causal_conv1d; print('causal-conv1d', causal_conv1d.__version__, 'on torch', torch.__version__)"
  fi
  touch "${TRAIN_VENV}/.arc3-ready"
  echo "venv built in $(( $(date +%s) - build_started ))s"
else
  echo "reusing venv ${TRAIN_VENV}"
fi
flock -u 9
__arc3_export_cuda_env
PLATYPUS_VENV="${TRAIN_VENV}"
if [ -n "${ARC3_PLATYPUS_VENV_OVERRIDE:-}" ]; then
  PLATYPUS_VENV="${ARC3_PLATYPUS_VENV_OVERRIDE}"
  echo "OVERRIDE: serving + driving from ${PLATYPUS_VENV}"
fi
export PLATYPUS_VENV TRAIN_VENV

# --- ports: deterministic base with a genuine free-port probe (co-resident
# jobs can hash to the same base and a listener there may be a neighbour's
# LIVE server — never reclaimed). Server A (base arm) on SERVER_PORT, server
# B (lora arm) on SERVER_PORT + 100; the +101 stride keeps successive
# candidates off each other's derived ports. -------------------------------
__arc3_port_free() {
  python3 -c 'import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()' "$1"
}
SERVER_PORT=$((50000 + SLURM_JOB_ID % 4000))
__arc3_ports_ok=0
for _ in $(seq 1 40); do
  if __arc3_port_free "${SERVER_PORT}" && __arc3_port_free $((SERVER_PORT + 100)); then
    __arc3_ports_ok=1
    break
  fi
  SERVER_PORT=$((SERVER_PORT + 101))
done
if [ "${__arc3_ports_ok}" != "1" ]; then
  echo "FATAL: no free server port pair found (base $((50000 + SLURM_JOB_ID % 4000)), 40 candidates tried)." >&2
  exit 1
fi
BASE_PORT="${SERVER_PORT}"
LORA_PORT=$((SERVER_PORT + 100))
echo "single-arm: lora server on port ${LORA_PORT} (GPU 0); eval dir ${EVAL_DIR}"

# --- vLLM servers (background). Both launch with runtime-LoRA updating on
# (launch_platypus_server.sh always enables it); only server B ever receives
# an adapter — the driver hot-loads it after health. setsid: each server owns
# a process group so teardown can signal the whole engine tree. -------------
SERVER_PIDS=()
SERVER_LOGS=()
__arc3_stop_servers() {
  local pid waited
  [ "${#SERVER_PIDS[@]}" -gt 0 ] || return 0
  for pid in "${SERVER_PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]}"; do
    waited=0
    while kill -0 "${pid}" 2>/dev/null && [ "${waited}" -lt 30 ]; do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
}
trap '__arc3_stop_servers' EXIT

for __arc3_arm in lora; do
  if [ "${__arc3_arm}" = "base" ]; then
    __arc3_port="${BASE_PORT}"; __arc3_gpu=0
  else
    __arc3_port="${LORA_PORT}"; __arc3_gpu=0
  fi
  __arc3_server_log="${EVAL_DIR}/inference_server_${__arc3_arm}.log"
  PLATYPUS_VENV="$PLATYPUS_VENV" \
  SERVER_MODEL="${ARC3_SERVER_MODEL_OVERRIDE:-Qwen/Qwen3.8-27B}" \
  SERVER_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
  SERVER_PORT="${__arc3_port}" \
  SERVER_GPU="${__arc3_gpu}" \
  SERVER_TENSOR_PARALLEL=1 \
  SERVER_MAX_MODEL_LEN="${ARC3_EVAL_MAX_MODEL_LEN:-98304}" \
  SERVER_MAX_LORA_RANK=16 \
  SERVER_LOG="${__arc3_server_log}" \
  setsid bash scripts/launch_platypus_server.sh &
  SERVER_PIDS+=($!)
  SERVER_LOGS+=("${__arc3_server_log}")
done

# --- wait for health (model list responds) before the driver starts.
# 100 min: a cold node pays venv sync + compile-cache build + weight load.
__arc3_health_deadline=$(( $(date +%s) + 6000 ))
for i in 0; do
  __arc3_port="${LORA_PORT}"
  until curl -sf "http://127.0.0.1:${__arc3_port}/models" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
      echo "server ${i} (port ${__arc3_port}) died during startup" >&2
      tail -40 "${SERVER_LOGS[$i]}" >&2 || true
      exit 1
    fi
    if [ "$(date +%s)" -ge "${__arc3_health_deadline}" ]; then
      echo "server ${i} (port ${__arc3_port}) not healthy before the deadline" >&2
      exit 1
    fi
    sleep 10
  done
  echo "server ${i} healthy on port ${__arc3_port}"
done

# --- the driver (platypus venv: it imports the harness like the rollout
# worker does, plus training.runtime_lora for the adapter export). Runs in
# the foreground; teardown fires on exit either way. ------------------------
# The || capture keeps set -e from killing the script before teardown and
# the status echo run.
__arc3_driver_status=0
# shellcheck disable=SC2086
"${PLATYPUS_VENV}/bin/python" scripts/eval_base_vs_lora.py \
  --config "${EVAL_CONFIG}" \
  --eval-dir "${EVAL_DIR}" \
  --lora-base-url "http://127.0.0.1:${LORA_PORT}" \
  --base-base-url "http://127.0.0.1:${LORA_PORT}" \
  --adapter-dir "${ADAPTER_DIR}" \
  ${ARC3_EVAL_DRIVER_ARGS:-} || __arc3_driver_status=$?

echo "driver exited with status ${__arc3_driver_status}; results in ${EVAL_DIR}"
echo "analyze with: python3 scripts/eval_base_vs_lora_analysis.py ${EVAL_DIR}"
exit "${__arc3_driver_status}"
