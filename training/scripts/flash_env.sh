#!/bin/bash
# Source this on a compute node to get the Flash runtime interpreter
# (ARC3_FLASH_PYTHON) and the process environment its engine and trainer need.
# Mirrors launch_platypus_server.sh / launch_routing_rl.sh: node-local compile
# caches, HF cache, and the CUDA toolkit env ullm stitches from pip wheels.
: "${ARC3_FLASH_REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ARC3_RL_CACHE_ROOT="${ARC3_RL_CACHE_ROOT:-/cache/shared/research/arc-rl}"
if [ ! -d "$(dirname "${ARC3_RL_CACHE_ROOT}")" ]; then
  echo "Shared node-local cache is not provisioned: ${ARC3_RL_CACHE_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi
umask 002
mkdir -p "${ARC3_RL_CACHE_ROOT}/flash"
if [ -z "${ARC3_FLASH_PYTHON:-}" ]; then
  ARC3_FLASH_PYTHON="$(bash "${ARC3_FLASH_REPO_ROOT}/scripts/prepare_flash_runtime.sh")"
  export ARC3_FLASH_PYTHON
fi
if [ ! -x "${ARC3_FLASH_PYTHON}" ]; then
  echo "Flash interpreter is not executable: ${ARC3_FLASH_PYTHON}" >&2
  return 1 2>/dev/null || exit 1
fi
export PYTHONPATH="${ARC3_FLASH_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V2_MODEL_RUNNER=1
# The trainer replays 32k-token sequences through 48 checkpointed layers;
# expandable segments keep the allocator from fragmenting around them.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${ARC3_RL_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export MAX_JOBS="${MAX_JOBS:-8}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
export ARC3_FLASH_COMPILE_CACHE_ROOT="${ARC3_RL_CACHE_ROOT}/flash/compile"
export VLLM_CACHE_ROOT="${ARC3_FLASH_COMPILE_CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${ARC3_FLASH_COMPILE_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${ARC3_FLASH_COMPILE_CACHE_ROOT}/inductor"
export XDG_CACHE_HOME="${ARC3_FLASH_COMPILE_CACHE_ROOT}/xdg"
__arc3_flash_key="$(printf '%s' "${ARC3_FLASH_PYTHON}" | sha256sum | cut -c1-16)"
export FLASHINFER_WORKSPACE_BASE="${ARC3_FLASH_COMPILE_CACHE_ROOT}/flashinfer-${__arc3_flash_key}"
mkdir -p "${HF_HUB_CACHE}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}"
# A throwaway interpreter computes ullm's CUDA env delta; importing ullm into
# the engine process itself would apply its event-loop patch.
__arc3_cuda_delta="$("${ARC3_FLASH_PYTHON}" - <<'PY'
import os
before = dict(os.environ)
from ullm.backends.config_local import set_env
set_env()
for key, value in os.environ.items():
    if before.get(key) != value:
        print(f"ARC3_ENV {key}={value}")
PY
)"
while IFS= read -r __arc3_line; do
  if [[ "${__arc3_line}" == 'ARC3_ENV '* ]]; then
    export "${__arc3_line#ARC3_ENV }"
  elif [ -n "${__arc3_line}" ]; then
    printf '%s\n' "${__arc3_line}" >&2
  fi
done <<< "${__arc3_cuda_delta}"
unset __arc3_cuda_delta __arc3_line __arc3_flash_key
