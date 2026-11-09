#!/bin/bash
#SBATCH --job-name=arc3-platypus-smoke
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
# --output must stay slurm-%j.out (not %x): external monitors key on the
# slurm-<jobid>.out filename. No #SBATCH --qos: an invalid QoS name fails
# submission outright and the right one varies by allocation — pass
# `sbatch --qos=...` when needed and export ARC3_QOS with the same value so
# continuations resubmit with it.
# 2-GPU platypus-stack debug smoke: GPU 0 serves Qwen3.8 (vllm from
# the platypus venv, runtime LoRA updating on), GPU 1 trains. The trainer
# runs connect-only against the fixed port in the debug config.
set -euo pipefail

# Default repo root: the checkout this script lives in. Under sbatch the
# script executes from a spooled copy, so fall back to the submit directory;
# ARC3_JOB_REPO_ROOT overrides both (continuations always pass it).
__arc3_self_root="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
if [ ! -f "${__arc3_self_root}/scripts/platypus_debug_smoke.sbatch" ]; then
  __arc3_self_root="${SLURM_SUBMIT_DIR:-${PWD}}"
fi
REPO_ROOT="${ARC3_JOB_REPO_ROOT:-${__arc3_self_root}}"
RUN_DIR="${ARC3_RUN_DIR:?set ARC3_RUN_DIR}"
if [ ! -f "${REPO_ROOT}/scripts/platypus_debug_smoke.sbatch" ]; then
  echo "FATAL: ${REPO_ROOT} is not an ARC3 checkout; export ARC3_JOB_REPO_ROOT." >&2
  exit 1
fi
cd "$REPO_ROOT"
REPO_ROOT="${PWD}"
RUN_DIR="$(realpath -m "${RUN_DIR}")"

# Kander's job env is required (TMPDIR on job-local disk, cluster paths): a
# node without it must fail the job, not fall back to host defaults.
. /etc/kander/job-env.sh
mkdir -p "$RUN_DIR"

# --- Auto-relaunch at end of job (ARC3_AUTO_RELAUNCH=1) --------------------
# No standing continuation sits in the queue: SLURM's end-of-job SIGTERM is
# trapped, and during teardown the segment resubmits a continuation of the
# SAME run directory (frozen_config.json pins the server port, the trainer
# restores checkpoints + controller state). Only scheduler endings chain
# (TIMEOUT/PREEMPTED/NODE_FAIL, with an elapsed-vs-limit fallback while
# slurmctld is still recording): a manual scancel or a crash ends the
# chain, so cancelled runs stay cancelled and crashes cannot loop.
# Kill switch at any time: touch "$RUN_DIR/STOP_AUTO_RELAUNCH".
if [ "${ARC3_AUTO_RELAUNCH:-0}" = "1" ] && [ -f "${RUN_DIR}/STOP_AUTO_RELAUNCH" ]; then
  if [ "${ARC3_CLEAR_STOP_ON_START:-0}" = "1" ]; then
    # A hand-submitted continuation (e.g. --dependency=afterany on the
    # running segment, with a different --time) takes over the chain: the
    # STOP file only had to keep the OLD segment's trap from chaining.
    rm -f "${RUN_DIR}/STOP_AUTO_RELAUNCH"
    echo "[auto-relaunch] STOP_AUTO_RELAUNCH cleared by this continuation; chaining resumes here."
  else
    echo "[auto-relaunch] STOP_AUTO_RELAUNCH present; not starting a segment."
    exit 0
  fi
fi
__arc3_got_endsig=0
__arc3_script_started_at="$(date +%s)"
trap '__arc3_got_endsig=1; exit 143' TERM
trap 'exit 130' INT
# Slurm TimeLimit (D-HH:MM:SS or HH:MM:SS) to seconds; empty for UNLIMITED.
__arc3_limit_seconds() {
  printf "%s" "$1" | awk -F"[-:]" "NF==4 {print ((\$1*24+\$2)*60+\$3)*60+\$4} NF==3 {print (\$1*60+\$2)*60+\$3}" || true
}
# Segment deadline for the trainer (training/segment_budget.py): it stops at
# the last iteration boundary whose state can still become durable, exits 75
# (see the trainer dispatch below) and is chained here. Default: the
# allocation's time limit less the same 300 s teardown margin the relaunch
# fallback uses; wrappers that budget their own segments export their own.
if [ -z "${ARC3_SEGMENT_DEADLINE_EPOCH:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  __arc3_limit_s="$(__arc3_limit_seconds "$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oE "TimeLimit=[0-9:-]+" | head -1 | cut -d= -f2 || true)")"
  if [ -n "${__arc3_limit_s}" ]; then
    export ARC3_SEGMENT_DEADLINE_EPOCH=$(( __arc3_script_started_at + __arc3_limit_s - 300 ))
    echo "[segment] deadline $(date -u -d "@${ARC3_SEGMENT_DEADLINE_EPOCH}" +%FT%TZ 2>/dev/null || echo "${ARC3_SEGMENT_DEADLINE_EPOCH}") from TimeLimit ${__arc3_limit_s}s"
  fi
fi
__arc3_maybe_relaunch() {
  [ "${ARC3_AUTO_RELAUNCH:-0}" = "1" ] || return 0
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  if [ -f "${RUN_DIR}/STOP_AUTO_RELAUNCH" ]; then
    echo "[auto-relaunch] STOP_AUTO_RELAUNCH present; ending the chain."
    return 0
  fi
  if [ "${TRAINER_RC:-0}" = "${ARC3_SEGMENT_YIELD_EXIT:-75}" ]; then
    echo "[auto-relaunch] trainer yielded its segment at a safe boundary; chaining."
  elif [ "${__arc3_got_endsig}" != "1" ]; then
    echo "[auto-relaunch] no end-of-job signal (trainer exit or crash); not chaining."
    return 0
  fi
  __arc3_job_info="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true)"
  __arc3_state="$(printf "%s" "${__arc3_job_info}" | grep -oE "JobState=[A-Z_]+" | head -1 | cut -d= -f2 || true)"
  __arc3_limit_raw="$(printf "%s" "${__arc3_job_info}" | grep -oE "TimeLimit=[0-9:-]+" | head -1 | cut -d= -f2 || true)"
  # A continuation inherits this segment's limit unless the run dir says
  # otherwise: NEXT_TIME_LIMIT (e.g. 12:00:00) lets a chain started with a
  # short, backfill-friendly segment grow to full-length segments later.
  __arc3_next_limit="$(cat "${RUN_DIR}/NEXT_TIME_LIMIT" 2>/dev/null | tr -d "[:space:]" || true)"
  # A yield is a deliberate stop from inside the allocation; the scheduler
  # state checks below only classify end-of-job signals.
  [ "${TRAINER_RC:-0}" = "${ARC3_SEGMENT_YIELD_EXIT:-75}" ] && __arc3_state="SEGMENT_YIELD"
  case "${__arc3_state}" in
    SEGMENT_YIELD)
      : ;;
    TIMEOUT|PREEMPTED|NODE_FAIL)
      echo "[auto-relaunch] job state ${__arc3_state}; chaining." ;;
    CANCELLED*)
      echo "[auto-relaunch] job cancelled; ending the chain."
      return 0 ;;
    *)
      # State not recorded yet during teardown: fall back to elapsed-vs-limit
      # (scontrol RunTime is unreliable here; use our own wall clock).
      __arc3_elapsed=$(( $(date +%s) - __arc3_script_started_at ))
      __arc3_limit_s="$(__arc3_limit_seconds "${__arc3_limit_raw}")"
      if [ -n "${__arc3_limit_s}" ] && [ "${__arc3_elapsed}" -ge $(( __arc3_limit_s - 300 )) ]; then
        echo "[auto-relaunch] state ${__arc3_state:-unknown} at ${__arc3_elapsed}s of ${__arc3_limit_s}s; treating as time limit and chaining."
      else
        echo "[auto-relaunch] state ${__arc3_state:-unknown} well before the limit (${__arc3_elapsed}s); treating as a deliberate stop."
        return 0
      fi ;;
  esac
  # The continuation must keep THIS allocation's resource shape, not the
  # file header's (an initial 5- or 8-GPU submission overrides the header):
  # recover it from scontrol, falling back to the header values.
  __arc3_gres="$(printf "%s" "${__arc3_job_info}" | grep -oE "TresPerNode=[^ ]+" | head -1 | cut -d= -f2 | sed -E 's#^gres[:/]##' || true)"
  __arc3_cpg="$(printf "%s" "${__arc3_job_info}" | grep -oE "CpusPerTres=[^ ]+" | head -1 | grep -oE "[0-9]+$" || true)"
  __arc3_mpg="$(printf "%s" "${__arc3_job_info}" | grep -oE "MemPerTres=[^ ]+" | head -1 | grep -oE "[0-9]+$" || true)"
  __arc3_qos="$(printf "%s" "${__arc3_job_info}" | grep -oE "QOS=[^ ]+" | head -1 | cut -d= -f2 || true)"
  case "${__arc3_qos}" in ""|"(null)"|"N/A") __arc3_qos="${ARC3_QOS:-}" ;; esac
  __arc3_resub=(
    --parsable
    --dependency="afterany:${SLURM_JOB_ID}"
    --account="${SLURM_JOB_ACCOUNT:-research}"
    --partition="${SLURM_JOB_PARTITION:-dev}"
    --time="${__arc3_next_limit:-${__arc3_limit_raw:-12:00:00}}"
    --job-name="${SLURM_JOB_NAME:-arc3-platypus}"
    --gres="${__arc3_gres:-gpu:b200:2}"
    --cpus-per-gpu="${__arc3_cpg:-8}"
    --mem-per-gpu="${__arc3_mpg:-96G}"
  )
  if [ -n "${__arc3_qos}" ]; then
    __arc3_resub+=(--qos="${__arc3_qos}")
  fi
  # Deadlines belong to one allocation; the next job computes its own.
  if __arc3_next="$(env -u TMPDIR -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u CUDA_VISIBLE_DEVICES -u ARC3_SEGMENT_DEADLINE_EPOCH \
    sbatch "${__arc3_resub[@]}" \
      --export="ALL,ARC3_JOB_REPO_ROOT=${REPO_ROOT},ARC3_RUN_DIR=${RUN_DIR},ARC3_CONFIG=${ARC3_CONFIG:-},ARC3_AUTO_RELAUNCH=1" \
      "${REPO_ROOT}/scripts/platypus_debug_smoke.sbatch" 2>/dev/null)"; then
    echo "[auto-relaunch] continuation ${__arc3_next} submitted for ${RUN_DIR}."
  else
    echo "[auto-relaunch] resubmit failed." >&2
  fi
}
__arc3_cleanup_build() {
  [ -n "${BUILD_SRC:-}" ] || return 0
  local build_path
  build_path="$(realpath -m -- "${BUILD_SRC}")"
  if [ "$(dirname -- "${build_path}")" != "$(realpath -m -- "${TMPDIR:?}")" ]; then
    echo "Refusing to remove unexpected build directory: ${BUILD_SRC}" >&2
    return 1
  fi
  case "$(basename -- "${build_path}")" in
    arc3-venv-build-*) rm -rf -- "${build_path}" ;;
    *) echo "Refusing to remove unexpected build directory: ${BUILD_SRC}" >&2; return 1 ;;
  esac
  BUILD_SRC=""
}
trap '__arc3_cleanup_build; __arc3_maybe_relaunch' EXIT
# ---------------------------------------------------------------------------

# Per-job ports so several smokes can share a node without the trainers
# cross-connecting to each other's servers (a fixed :58900 once let a new
# run hot-load its adapter into a neighbour's vLLM). Server i listens on
# SERVER_PORT + 100*i; each collector's gateway on its server port + 5000
# (training/inference.py).
# One node-local cache root for EVERY RL run on the node, shared across
# users: /cache/shared/research/arc-rl (group research, setgid) already holds
# compile/, huggingface/ on every RL node. umask 002 keeps what we create
# group-writable so teammates' jobs can reuse and refresh it.
umask 002
# The node-local cache root is the SHARED research directory (one venv,
# one weight cache, one compile cache per node for every RL run). There is
# deliberately no per-user fallback: a node without it is misprovisioned,
# and silently building a private cache there would hide that.
ARC3_RL_CACHE_ROOT="${ARC3_RL_CACHE_ROOT:-/cache/shared/research/arc-rl}"
if ! { [ -d "$(dirname "${ARC3_RL_CACHE_ROOT}")" ] && mkdir -p "${ARC3_RL_CACHE_ROOT}" 2>/dev/null && [ -w "${ARC3_RL_CACHE_ROOT}" ]; }; then
  echo "FATAL: node-local cache root ${ARC3_RL_CACHE_ROOT} is missing or not writable on $(hostname)." >&2
  echo "       /cache/shared/research must be provisioned (group research, setgid) on every RL node; ask the admins, or submit with --exclude=$(hostname)." >&2
  exit 1
fi
echo "rl cache root: ${ARC3_RL_CACHE_ROOT} (shared)"
export ARC3_RL_CACHE_ROOT
export ARC3_COMPILE_CACHE_ROOT="${ARC3_COMPILE_CACHE_ROOT:-${ARC3_RL_CACHE_ROOT}/compile}"
# The TRAINER's JIT caches (fla Triton kernels, inductor, flash-attn-4's
# CuTe DSL under XDG) must be node-local too: Triton's default is
# ~/.triton/cache on NFS home, where one kernel compile is dozens of
# metadata round-trips -- r31/r32 sat for 10+ minutes in their first
# backward pass with the GPU idle. The server launcher sets its own
# (-platypus) variants of these; the trainer inherits these.
export TRITON_CACHE_DIR="${ARC3_COMPILE_CACHE_ROOT}/triton-trainer"
export TORCHINDUCTOR_CACHE_DIR="${ARC3_COMPILE_CACHE_ROOT}/inductor-trainer"
export XDG_CACHE_HOME="${ARC3_COMPILE_CACHE_ROOT}/xdg-trainer"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}"
# Weights: downloaded from the Hub into the shared node-local HF cache.
export HF_HOME="${ARC3_RL_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"

# Trainer/server use the ARC-3 workspace; ARC-2 workers have a separate lock
# and interpreter because both harnesses define duck, platypus, and taaf.
bash "${REPO_ROOT}/scripts/sync_platypus.sh"
SRC_CONFIG="${ARC3_CONFIG:-configs/ppo_2gpu_arc3_games25_platypus.json}"
SHAPE_CONFIG="${SRC_CONFIG}"
if [ -f "${RUN_DIR}/frozen_config.json" ]; then
  SHAPE_CONFIG="${RUN_DIR}/frozen_config.json"
fi
SERVER_ENVIRONMENT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["inference"].get("environment", "arc3"))' "${SHAPE_CONFIG}")"
case "${SERVER_ENVIRONMENT}" in arc3|arc2) ;; *) echo "FATAL: unknown training environment ${SERVER_ENVIRONMENT}" >&2; exit 1 ;; esac
export SERVER_ENVIRONMENT
SERVER_MULTIMODAL_CONTEXT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["inference"].get("agent_multimodal_context", ""))' "${SHAPE_CONFIG}")"
export SERVER_MULTIMODAL_CONTEXT
VENV_ROOT="${VENV_ROOT:-${ARC3_RL_CACHE_ROOT}/venvs}"
# uv's cache is PER-USER per cluster guidance; only the venvs themselves are
# shared (hash-keyed, effectively immutable once .arc3-ready is stamped).
export UV_CACHE_DIR="/cache/users/$(id -u)/uv"
mkdir -p "${UV_CACHE_DIR}"
# The venv key covers everything that determines its CONTENT: the lock, the
# platypus submodule commit (the workspace members' source — the lock does
# not hash local sources), the recipe, and uv itself. Members are installed
# non-editable (below) so the venv never resolves imports back into whichever
# clone happened to build it — job 2297392 imported another checkout's duck
# through a reused venv's editable path.
PLATYPUS_SHA="$(git -C platypus rev-parse HEAD)"
ARC2_PLATYPUS_SHA="$(git -C arc2-platypus rev-parse HEAD)"
export ARC3_FAST_KERNELS="${ARC3_FAST_KERNELS:-1}"
# The fast-kernel packages are installed OUTSIDE uv.lock, so the flag and
# their exact recipe are part of the key: a venv built with
# ARC3_FAST_KERNELS=0 must never be reused by a default job.
FAST_KERNEL_RECIPE="flash-linear-attention==0.5.1 fla-core==0.5.1 causal-conv1d==1.6.2.post1"
CUTE_CU13_RECIPE="nvidia-cutlass-dsl-libs-cu13==4.5.2"
TRAIN_EXTRAS=(--extra harness --extra fa4)
LOCK_HASH="$({ sha256sum uv.lock environments/arc2-worker/uv.lock environments/arc2-worker/pyproject.toml; printf 'platypus=%s\narc2-platypus=%s\n' "${PLATYPUS_SHA}" "${ARC2_PLATYPUS_SHA}"; printf 'recipe=v9\n'; printf 'train_extras=%s\n' "${TRAIN_EXTRAS[*]}"; printf 'fast_kernels=%s %s\n' "${ARC3_FAST_KERNELS}" "${FAST_KERNEL_RECIPE}"; printf 'cute_overlay=%s\n' "${CUTE_CU13_RECIPE}"; uv --version; } | sha256sum | cut -c1-16)"
TRAIN_VENV="${VENV_ROOT}/${LOCK_HASH}"
# Cap native source builds (causal-conv1d, sdist fallbacks, runtime JIT) to
# the allocation instead of every core on the node.
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export MAX_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL}"
# Both locked CuTe library wheels own the same package paths. CUDA 13's
# implementation must win over the base wheel's unsupported-toolkit stubs.
# Apply only during a new environment build, before its ready marker.
__arc3_install_cute_cuda13() {
  local target_venv="$1"
  env -u VIRTUAL_ENV uv pip install --python "${target_venv}/bin/python" \
    --no-deps --reinstall-package nvidia-cutlass-dsl-libs-cu13 "${CUTE_CU13_RECIPE}"
  "${target_venv}/bin/python" - <<'PY'
import hashlib
from importlib.metadata import distribution

package = distribution("nvidia-cutlass-dsl-libs-cu13")
prefix = "nvidia_cutlass_dsl/python_packages/cutlass/"
expected = {
    "cute/experimental/__init__.py": "a50a8589ecfc4869b1963420a7082d259284f794fe2f2c470accde3b07025f6f",
    "_mlir/_mlir_libs/_cutlass_ir.cpython-312-x86_64-linux-gnu.so": "73b760621e35910305e7bdf8f4c2c0d928c10527a243f8f11a76046edba4f6d8",
}
for relative, digest in expected.items():
    path = package.locate_file(prefix + relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"CuTe CUDA 13 package payload differs: {path}")
print(f"verified CuTe CUDA 13 {package.version} implementation and binary")
PY
}
# CUDA/NVCC env for driver-only nodes: ullm's set_env stitches the pip
# nvidia-cuda-nvcc wheel into a usable CUDA_HOME. The delta is captured via
# command substitution so a failing python aborts the job — a broken process
# substitution would feed an empty loop and "succeed" under set -e.
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
# Several jobs can start on one node at once; the build runs under a lock
# so only one builds and the rest wait for the ready marker.
mkdir -p "${VENV_ROOT}"
exec 9>"${VENV_ROOT}/${LOCK_HASH}.lock"
flock -w 3600 9
if [ ! -f "${TRAIN_VENV}/.arc3-ready" ]; then
  echo "building venv ${TRAIN_VENV} from uv.lock"
  build_started=$(date +%s)
  # Build from a pristine job-local copy of the sources: the setuptools
  # wheel builds for the workspace members write build/ and *.egg-info INTO
  # the project directory, so two nodes cold-building concurrently from the
  # shared NFS checkout corrupt each other (jobs 2298462/2298464). git
  # archive gives exactly the committed tree, nothing more. It takes the whole
  # monorepo because uv's path sources are sibling directories of training/.
  BUILD_SRC="$(mktemp -d "${TMPDIR:?TMPDIR must come from /etc/kander/job-env.sh}/arc3-venv-build-${SLURM_JOB_ID}.XXXX")"
  git -C "$(git rev-parse --show-toplevel)" archive HEAD | tar -x -C "${BUILD_SRC}"
  (cd "${BUILD_SRC}/training" && env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="${TRAIN_VENV}" uv sync --locked "${TRAIN_EXTRAS[@]}" --no-editable)
  __arc3_install_cute_cuda13 "${TRAIN_VENV}"
  __arc3_cleanup_build
  if [ "${ARC3_FAST_KERNELS}" != "0" ]; then
    # Trainer fast kernels (fla/DeltaNet + causal-conv1d). If a matching
    # prebuilt wheel is unavailable, compile once per venv hash. nvcc comes
    # from the nvidia-cuda-nvcc wheel ullm already
    # ships (the same CUDA env delta the server launcher uses).
    __arc3_export_cuda_env
    env -u VIRTUAL_ENV uv pip install --python "${TRAIN_VENV}/bin/python" \
      flash-linear-attention==0.5.1 fla-core==0.5.1
    # causal-conv1d is a C++/CUDA extension compiled against a specific torch
    # ABI. Prefer the prebuilt wheel for the installed torch on the shared
    # root; otherwise build from source with uv's cache bypassed -- uv keys
    # an sdist build on the source alone, so a build cached from another
    # torch version would be reused and segfault in causal_conv1d_fwd.
    # uv.lock can contain several platform-specific torch versions; select
    # the one actually installed by uv sync in this training environment.
    TORCH_VERSION="$("${TRAIN_VENV}/bin/python" -c 'from importlib.metadata import version; print(version("torch").split("+", 1)[0])')"
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
  "${TRAIN_VENV}/bin/python" -c "from training.verify_fa4 import require_fa4_distribution; print('flash-attn-4', require_fa4_distribution())"
  touch "${TRAIN_VENV}/.arc3-ready"
  echo "venv built in $(( $(date +%s) - build_started ))s"
else
  echo "reusing venv ${TRAIN_VENV}"
fi
flock -u 9
# The trainer's fast-kernel/JIT path needs the CUDA env on EVERY job, not
# only on the one that cold-built the venv (idempotent after a cold build).
__arc3_export_cuda_env
PLATYPUS_VENV="${TRAIN_VENV}"
# Bisection only: serve and collect from a different environment (e.g. the
# 0.25.1 platypus venv) while the trainer stays on this lock.
if [ -n "${ARC3_PLATYPUS_VENV_OVERRIDE:-}" ]; then
  PLATYPUS_VENV="${ARC3_PLATYPUS_VENV_OVERRIDE}"
  echo "BISECT: server + rollout worker from ${PLATYPUS_VENV}"
fi
WORKER_VENV="${PLATYPUS_VENV}"
if [ "${SERVER_ENVIRONMENT}" = "arc2" ]; then
  WORKER_VENV="${VENV_ROOT}/${LOCK_HASH}-arc2-worker"
  exec 8>"${VENV_ROOT}/${LOCK_HASH}-arc2-worker.lock"
  flock -w 3600 8
  if [ ! -f "${WORKER_VENV}/.arc2-ready" ]; then
    BUILD_SRC="$(mktemp -d "${TMPDIR:?}/arc3-venv-build-${SLURM_JOB_ID}.XXXX")"
    git -C "$(git rev-parse --show-toplevel)" archive HEAD | tar -x -C "${BUILD_SRC}"
    (cd "${BUILD_SRC}/training" && env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="${WORKER_VENV}" uv sync --project environments/arc2-worker --locked --no-editable)
    __arc3_install_cute_cuda13 "${WORKER_VENV}"
    __arc3_cleanup_build
    touch "${WORKER_VENV}/.arc2-ready"
  fi
  flock -u 8
fi
export PLATYPUS_VENV TRAIN_VENV WORKER_VENV

# Weights: HF downloads the pinned revision straight into the node-local
# cache (HF_HUB_CACHE above) the first time a node serves this model; no
# mirror, no copying.

LAUNCH_CONFIG="${RUN_DIR}/launch_config.json"
# Server topology comes from the governing config (the frozen one on resume):
# inference.server_gpus lists one GPU id per vLLM server.
# official_eval.server_gpus extend the sequence: eval server j is server
# T+j on SERVER_PORT + 100*(T+j), which training.official_eval derives the
# same way (_eval_server_base_urls), so the held-out eval process connects
# to a server this job launched.
SERVER_GPU_LIST="$(python3 -c 'import json, sys
cfg = json.load(open(sys.argv[1]))
servers = list(cfg["inference"].get("server_gpus") or ["0"])
official_eval = cfg.get("official_eval") or {}
if official_eval.get("enabled"):
    servers += [str(gpu) for gpu in official_eval.get("server_gpus") or []]
print(" ".join(servers))' "${SHAPE_CONFIG}")"
read -r -a SERVER_GPUS <<< "${SERVER_GPU_LIST}"
NUM_SERVERS="${#SERVER_GPUS[@]}"
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
# Deterministic base port with a genuine free-port probe: co-resident jobs
# can hash to the same base, and a listener there may be a neighbour's LIVE
# server — it is never reclaimed (launch_platypus_server.sh fails fast
# instead), a different base is chosen. Every server port (+100*i) and
# gateway port (+100*i+5000) must be free; the +101 stride keeps successive
# candidates off each other's derived ports.
SERVER_PORT=$((50000 + SLURM_JOB_ID % 4000))
__arc3_ports_ok=0
for _ in $(seq 1 40); do
  __arc3_ports_ok=1
  for (( i = 0; i < NUM_SERVERS; i++ )); do
    if ! __arc3_port_free $((SERVER_PORT + 100 * i)) || \
       ! __arc3_port_free $((SERVER_PORT + 100 * i + 5000)); then
      __arc3_ports_ok=0
      break
    fi
  done
  if [ "${__arc3_ports_ok}" = "1" ]; then
    break
  fi
  SERVER_PORT=$((SERVER_PORT + 101))
done
if [ "${__arc3_ports_ok}" != "1" ]; then
  echo "FATAL: no free server port block found (base $((50000 + SLURM_JOB_ID % 4000)), 40 candidates tried)." >&2
  exit 1
fi
if [ -f "${RUN_DIR}/frozen_config.json" ]; then
  # Resume: ports are allocation-local, not run identity. The old design
  # reused the run's ORIGINAL port, but a resumed segment can land on a
  # node where that port is already taken — seed1337's continuation died
  # binding its historical gateway port. Every segment derives fresh ports
  # from THIS job id, and the frozen config's base_url is rewritten so the
  # trainer, server, gateway and worker all agree. The venv is keyed on the
  # CURRENT lock/submodule, so platypus_venv is repointed at the build this
  # leg made — worker and trainer must share one harness build. The rewrite
  # is atomic (temp file + fsync + rename): an interrupted leg must never
  # leave a truncated frozen_config.json behind.
  python3 - "${RUN_DIR}/frozen_config.json" "${SERVER_PORT}" "${WORKER_VENV}" <<'PY'
import json, os, sys, tempfile
path, port, venv = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(path))
cfg["inference"]["base_url"] = f"http://127.0.0.1:{port}"
cfg["inference"]["platypus_venv"] = venv
from pathlib import Path
from training.server_endpoints import configure_server_log_paths
configure_server_log_paths(cfg, Path(path).parent)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
try:
    with os.fdopen(fd, "w") as handle:
        json.dump(cfg, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
except BaseException:
    os.unlink(tmp)
    raise
PY
  echo "resuming ${RUN_DIR}: server port ${SERVER_PORT} + venv rewritten into frozen_config.json"
fi
python3 - "$SRC_CONFIG" "$LAUNCH_CONFIG" "$SERVER_PORT" "$WORKER_VENV" <<'PY'
import json, sys
src, dst, port, venv = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(src))
cfg["inference"]["base_url"] = f"http://127.0.0.1:{port}"
# The selected harness owns its node-local worker interpreter.
cfg["inference"]["platypus_venv"] = venv
from pathlib import Path
from training.server_endpoints import configure_server_log_paths
configure_server_log_paths(cfg, Path(dst).parent)
json.dump(cfg, open(dst, "w"), indent=2)
PY
echo "server base port ${SERVER_PORT} (${NUM_SERVERS} server(s), +100 each, gateways +5000), config ${LAUNCH_CONFIG}"

# Freeze once, before any server starts. On resume this reuses source.tar:
# both the server launcher/middleware and trainer must come from that tree,
# even when the live checkout has changed since the run began.
export ARC3_RUN_DIR="${RUN_DIR}"
ARC3_FROZEN_REPO_ROOT="$("${TRAIN_VENV}/bin/python" -m training.launch_frozen \
  --config "${LAUNCH_CONFIG}" --prepare-only)"
export ARC3_FROZEN_REPO_ROOT
export ARC3_ORIGINAL_REPO_ROOT="${REPO_ROOT}"
export ARC3_FROZEN_CONFIG_PATH="${RUN_DIR}/frozen_config.json"
export ARC3_RUNTIME_FINGERPRINT_PATH="${RUN_DIR}/runtime_fingerprint.json"
export PYTHONPATH="${ARC3_FROZEN_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export FLA_TILELANG="${FLA_TILELANG:-0}"
export FLA_CONV_BACKEND="${FLA_CONV_BACKEND:-triton}"
# A local model override is relative to the submitting checkout, not the
# frozen source directory we are about to enter.
if [ -d "${ARC3_SERVER_MODEL_OVERRIDE:-}" ]; then
  export ARC3_SERVER_MODEL_OVERRIDE="$(realpath "${ARC3_SERVER_MODEL_OVERRIDE}")"
fi
cd "${ARC3_FROZEN_REPO_ROOT}"

# Validate the effective allocation with the frozen code and config, before
# starting any GPU process. Environment overrides must not overlap a trainer
# or another server, and the served model must match the training config.
SERVER_SHAPE="$("${TRAIN_VENV}/bin/python" - "${ARC3_FROZEN_CONFIG_PATH}" <<'PY'
import os
import sys
from training.config import (
    ExperimentConfig, evaluation_inference_config, max_served_sequence_tokens,
)

cfg = ExperimentConfig.load(sys.argv[1])
servers = list(cfg.inference.server_gpus or ["0"])
training_context = max_served_sequence_tokens(cfg.inference) or 40960
context_windows = [training_context] * len(servers)
if cfg.official_eval.enabled:
    servers += cfg.official_eval.server_gpus
    eval_context = (
        max_served_sequence_tokens(evaluation_inference_config(cfg)) or training_context
    )
    context_windows += [eval_context] * len(cfg.official_eval.server_gpus)
if len(servers) == 1:
    servers = [os.environ.get("ARC3_SERVER_GPUS") or servers[0]]
tp = int(os.environ.get("ARC3_SERVER_TENSOR_PARALLEL") or "1")
training_gpus = set(cfg.training.device_map_gpus)
if cfg.training.weight_mode == "full_weight":
    training_gpus.update(cfg.training.full_weight.device_ids)
for replica in cfg.training.worker_device_map_gpus:
    training_gpus.update(replica)
if not training_gpus and str(cfg.training.device).startswith("cuda"):
    training_gpus.add(int(str(cfg.training.device).partition(":")[2] or "0"))
seen = set(training_gpus)
for server in servers:
    gpus = [int(gpu) for gpu in str(server).split(",")]
    if tp < 1 or len(gpus) != tp or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("Effective server GPU list must contain one distinct nonnegative GPU per tensor-parallel rank")
    overlap = seen.intersection(gpus)
    if overlap:
        raise ValueError(f"Effective server GPUs must be disjoint from trainer/eval claims; GPUs {sorted(overlap)} are claimed twice")
    seen.update(gpus)
for value in (cfg.training.model_name, cfg.training.model_revision,
              cfg.inference.model_id or cfg.training.model_name,
              cfg.training.lora.r, " ".join(servers),
              " ".join(map(str, context_windows)),
              ",".join(str(gpu) for gpu in range(max(seen) + 1)),
              cfg.training.weight_mode,
              cfg.training.score_centering_top_k or 0,
              ",".join(map(str, cfg.training.full_weight.device_ids)) or "-"):
    print(value)
PY
)"
{
  IFS= read -r SERVER_MODEL
  IFS= read -r SERVER_REVISION
  IFS= read -r SERVER_CONFIGURED_NAME
  IFS= read -r SERVER_MAX_LORA_RANK
  read -r -a SERVER_GPUS
  read -r -a SERVER_CONTEXT_WINDOWS
  IFS= read -r DEFAULT_VISIBLE_GPUS
  IFS= read -r SERVER_WEIGHT_MODE
  IFS= read -r SERVER_SCORE_CENTERING_TOP_K
  IFS= read -r FULL_WEIGHT_TRAINER_GPUS
} <<< "${SERVER_SHAPE}"
if [ "${#SERVER_GPUS[@]}" -ne "${NUM_SERVERS}" ]; then
  echo "FATAL: frozen server count changed after reserving the port block." >&2
  exit 1
fi
export SERVER_MODEL SERVER_REVISION SERVER_MAX_LORA_RANK SERVER_WEIGHT_MODE SERVER_SCORE_CENTERING_TOP_K

# --- vLLM server(s) (background), logs where the collector expects them.
# With one server ARC3_SERVER_GPUS can override its configured GPU list,
# with ARC3_SERVER_TENSOR_PARALLEL (default 1) ranks
# over them; with a multi-server config, server i serves from
# server_gpus[i] on SERVER_PORT + 100*i. The trainer addresses its GPUs by
# index from the config (device / device_map_gpus / worker_device_map_gpus)
# within ARC3_VISIBLE_GPUS (default "0,1").
# An 8-GPU job: sbatch --gres=gpu:b200:8 with ARC3_SERVER_GPUS=0,1,2,3
# ARC3_SERVER_TENSOR_PARALLEL=4 ARC3_VISIBLE_GPUS=0,1,2,3,4,5,6,7 and a
# config whose trainer lives on cuda:4..7.
SERVER_PIDS=()
SERVER_LOGS=()
__arc3_stop_servers() {
  local pid stop_deadline
  local children=("${SERVER_PIDS[@]}")
  if [ -n "${TRAINER_PID:-}" ]; then children+=("${TRAINER_PID}"); fi
  [ "${#children[@]}" -gt 0 ] || return 0
  for pid in "${children[@]}"; do
    # The HTTP host drains its draws before shutting down the engine children.
    kill -TERM "${pid}" 2>/dev/null || true
  done
  stop_deadline=$((SECONDS + 30))
  for pid in "${children[@]}"; do
    while kill -0 "${pid}" 2>/dev/null && [ "${SECONDS}" -lt "${stop_deadline}" ]; do
      sleep 1
    done
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
}
trap '__arc3_stop_servers; __arc3_cleanup_build; __arc3_maybe_relaunch' EXIT
for (( i = 0; i < NUM_SERVERS; i++ )); do
  if [ "${NUM_SERVERS}" -eq 1 ]; then
    __arc3_server_gpu="${SERVER_GPUS[$i]}"
    __arc3_server_log="${RUN_DIR}/inference_server.log"
  else
    __arc3_server_gpu="${SERVER_GPUS[$i]}"
    __arc3_server_log="${RUN_DIR}/inference_server_${i}.log"
  fi
  # setsid: each server owns a process group, so teardown can signal the
  # whole engine tree (vLLM forks workers), wait bounded, then escalate.
  PLATYPUS_VENV="$PLATYPUS_VENV" \
  SERVER_SERVED_NAME="${SERVER_SERVED_NAME:-${SERVER_CONFIGURED_NAME}}" \
  SERVER_MODEL="${ARC3_SERVER_MODEL_OVERRIDE:-${SERVER_MODEL}}" \
  SERVER_PORT="$((SERVER_PORT + 100 * i))" \
  SERVER_GPU="${__arc3_server_gpu}" \
  SERVER_TENSOR_PARALLEL="${ARC3_SERVER_TENSOR_PARALLEL:-1}" \
  SERVER_MAX_MODEL_LEN="${SERVER_CONTEXT_WINDOWS[$i]}" \
  SERVER_LOG="${__arc3_server_log}" \
  setsid bash "${ARC3_FROZEN_REPO_ROOT}/scripts/launch_platypus_server.sh" &
  SERVER_PIDS+=($!)
  SERVER_LOGS+=("${__arc3_server_log}")
done

# --- wait for health (model list responds) before the trainer starts.
# 100 min: a cold node pays venv sync + compile-cache build + weight load.
__arc3_health_deadline=$(( $(date +%s) + 6000 ))
for (( i = 0; i < NUM_SERVERS; i++ )); do
  __arc3_port=$((SERVER_PORT + 100 * i))
  until "${PLATYPUS_VENV}/bin/python" -m training.server_endpoints \
    "http://127.0.0.1:${__arc3_port}/models" > /dev/null 2>&1; do
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

# --- training (same venv as the server) on the config's GPUs.
export CUDA_VISIBLE_DEVICES="${ARC3_VISIBLE_GPUS:-${DEFAULT_VISIBLE_GPUS}}"
# The snapshot was prepared before serving; do not freeze a second time
# (especially when ARC3_FORCE_SNAPSHOT was requested for this segment).
if [ "${SERVER_WEIGHT_MODE}" = "full_weight" ]; then
  IFS=, read -r -a __arc3_trainer_gpus <<< "${FULL_WEIGHT_TRAINER_GPUS}"
  CUDA_VISIBLE_DEVICES="${FULL_WEIGHT_TRAINER_GPUS}" setsid "${TRAIN_VENV}/bin/python" \
    -m training.full_weight_launch --standalone --nnodes=1 \
    --nproc-per-node="${#__arc3_trainer_gpus[@]}" --max-restarts=0 \
    --module training.full_weight_entrypoint --config "${ARC3_FROZEN_CONFIG_PATH}" &
else
  setsid "${TRAIN_VENV}/bin/python" -m training.cli \
    --config "${ARC3_FROZEN_CONFIG_PATH}" &
fi
TRAINER_PID=$!
# Exit 75 (EX_TEMPFAIL) is the trainer's segment yield: its state is durable
# and the run directory wants a continuation (training/shutdown.py).
ARC3_SEGMENT_YIELD_EXIT=75
TRAINER_RC=0
wait "${TRAINER_PID}" || TRAINER_RC=$?
if [ "${TRAINER_RC}" = "${ARC3_SEGMENT_YIELD_EXIT}" ]; then
  echo "[segment] trainer yielded at a safe boundary (exit ${TRAINER_RC}); resume this run directory to continue."
fi
exit "${TRAINER_RC}"
