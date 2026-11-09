#!/bin/bash
# Native AsyncLLMEngine launcher. The trainer gateway receives integer-token
# generations; the service owns sampling parameters and the policy controls
# (runtime LoRA, NCCL full weights, or Flash's merged policy weights).
# PLATYPUS_VENV pins vLLM 0.25.1, or fork 3bda1ba09 for Flash. No OpenAI API.
# Required: PLATYPUS_VENV, SERVER_MODEL, SERVER_PORT, SERVER_REVISION (non-Flash),
# ARC3_RL_CACHE_ROOT. Optional engine tuning remains in SERVER_EXTRA_ARGS;
# SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS (default 248058, <tool_call>) is the one
# id set a request may bias.
set -euo pipefail

case "${SERVER_WEIGHT_MODE:-lora}" in
  lora|full_weight|flash) ;;
  *) echo "launch_platypus_server: unsupported SERVER_WEIGHT_MODE" >&2; exit 1 ;;
esac

: "${PLATYPUS_VENV:?set PLATYPUS_VENV}"
: "${SERVER_MODEL:?set SERVER_MODEL}"
if [ "${SERVER_WEIGHT_MODE:-lora}" = "flash" ]; then
  if [ ! -d "${SERVER_MODEL}" ]; then
    echo "launch_platypus_server: flash requires SERVER_MODEL as a local checkpoint directory" >&2
    exit 1
  fi
else
  : "${SERVER_REVISION:?set SERVER_REVISION}"
fi
: "${SERVER_PORT:?set SERVER_PORT}"

export CUDA_VISIBLE_DEVICES="${SERVER_GPU:-0}"

# The KV cache is GPU-only, at all times, for rollouts and eval alike (owner
# decision, 2026-09-04: "Make sure CPU KV cache is off as well at all
# times in rollout and training"). The pinned vLLM (0.25.1, V1 engine) has
# no CPU swap space any more, its KV offloading (--kv-offloading-size) and
# KV transfer default to off, and weight offload (--cpu-offload-gb) to 0;
# this launcher refuses pass-through offload arguments. Flash alone puts its
# frozen PLE table in host memory below (Engram); KV remains GPU-only.
if [ -n "${SERVER_EXTRA_ARGS:-}" ] && printf "%s" "${SERVER_EXTRA_ARGS}" | grep -qE -- "--(kv-offloading-size|kv-offloading-backend|kv-transfer-config|cpu-offload-gb|cpu-offload-params|engram-config|swap-space|kv_offloading_size|cpu_offload_gb|cpu_offload_params|engram_config|swap_space)"; then
  echo "launch_platypus_server: SERVER_EXTRA_ARGS would enable a CPU KV cache / offload (${SERVER_EXTRA_ARGS}); the KV cache is GPU-only, refusing to serve" >&2
  exit 1
fi
# Drafting is a run setting (inference.speculative_decoding, through
# SERVER_SPECULATIVE_CONFIG) that /runtime attests; never a pass-through.
if [ -n "${SERVER_EXTRA_ARGS:-}" ] && printf "%s" "${SERVER_EXTRA_ARGS}" | grep -qE -- "--(speculative-config|speculative_config)"; then
  echo "launch_platypus_server: SERVER_EXTRA_ARGS would configure speculative decoding (${SERVER_EXTRA_ARGS}); set inference.speculative_decoding instead" >&2
  exit 1
fi
if [ -n "${SERVER_SPECULATIVE_CONFIG:-}" ] && [ "${SERVER_WEIGHT_MODE:-lora}" != "flash" ]; then
  echo "launch_platypus_server: SERVER_SPECULATIVE_CONFIG is a Flash serving setting" >&2
  exit 1
fi
# Never reclaim the port: a listener here can be a co-resident job's LIVE
# server (ports are hashed from job ids and can collide — the old
# kill-any-listener logic could SIGKILL a neighbour's vLLM). The batch
# script probes for a genuinely free port before choosing; if ours was taken
# since, fail fast so the job aborts instead of serving into a conflict.
if ! python3 -c 'import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()' "${SERVER_PORT}"; then
  echo "launch_platypus_server: port ${SERVER_PORT} is already in use (a live co-resident server?); refusing to serve" >&2
  exit 1
fi

ARGS=(
  --model "${SERVER_MODEL}"
  --served-model-name "${SERVER_SERVED_NAME:-Qwen/Qwen3.8-27B}"
  --host "${SERVER_HOST:-127.0.0.1}"
  --port "${SERVER_PORT}"
  --max-model-len "${SERVER_MAX_MODEL_LEN:-32768}"
  --tensor-parallel-size "${SERVER_TENSOR_PARALLEL:-1}"
  --enable-prefix-caching
  # PPO reproduces the admitted token bias before scoring. Return logprobs
  # after processors; ordinary training requests have neutral processing.
  --generation-config vllm
  --override-generation-config '{}'
  --dtype bfloat16
  --logprobs-mode processed_logprobs
  --score-centering-top-k "${SERVER_SCORE_CENTERING_TOP_K:-0}"
  # The only token ids a request may bias: the summary's ban on <tool_call>
  # (the Duck's QWEN_TOOL_CALL_TOKEN_ID, the recipes'
  # agent_summary_forbidden_token_ids). The server refuses to start unless
  # each decodes to <tool_call> under the served tokenizer. Every model served
  # here shares the Qwen3.6/3.8 vocabulary; set it empty to bias nothing.
  --summary-forbidden-token-ids "${SERVER_SUMMARY_FORBIDDEN_TOKEN_IDS-248058}"
  # No CPU KV cache (owner decision): the KV pool is GPU-only. vLLM's CPU
  # offload (--kv-offloading-size) defaults to disabled; never enable it.
)
case "${SERVER_WEIGHT_MODE:-lora}" in
  # flash: no LoRA. The learner's rank 0 broadcasts merged policy weights over
  # NCCL into the served weights (/weights/*, training/flash_policy_weights.py).
  full_weight|flash) ARGS+=(--weight-transfer-config '{"backend":"nccl"}') ;;
  lora)
    # Two slots permit loading the replacement before retiring its old ID.
    # Each draw carries its immutable LoRARequest; aliases change only drained.
    ARGS+=(--enable-lora --max-lora-rank "${SERVER_MAX_LORA_RANK:-16}" --max-loras "${SERVER_MAX_LORAS:-2}")
    ;;
esac
case "${SERVER_ENVIRONMENT:-arc3}" in
  arc2)
    ARGS+=(--limit-mm-per-prompt '{"video":0}' --mm-encoder-attn-backend FLASH_ATTN)
    ;;
  arc3)
    case "${SERVER_MULTIMODAL_CONTEXT:-}" in
      current_grid|level_start)
        ARGS+=(--limit-mm-per-prompt '{"video":0}' --mm-encoder-attn-backend FLASH_ATTN)
        ;;
      "") ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0}') ;;
      *) echo "launch_platypus_server: unsupported SERVER_MULTIMODAL_CONTEXT" >&2; exit 1 ;;
    esac
    ;;
  *) echo "launch_platypus_server: unsupported SERVER_ENVIRONMENT" >&2; exit 1 ;;
esac
# A local checkpoint dir (ARC3_SERVER_MODEL_OVERRIDE: a merged LoRA, an FP8 copy)
# has no HF revision; the HF id keeps the pinned one. The served name stays the
# HF id so agents, configs and the trainer address one model name.
if [ ! -d "${SERVER_MODEL}" ]; then
  ARGS+=(--revision "${SERVER_REVISION}" --tokenizer-revision "${SERVER_REVISION}")
fi
# GDN (linear-attention) prefill kernel: Triton/FLA by default, the kernel
# family the trainer differentiates through (fla) and the one 0.25.1 served
# with. vLLM 0.27's "auto" picks FlashInfer's kernel on Blackwell, which
# doubled the trainer/server logprob mismatch (mean |d| 0.02 -> 0.05) and
# put every run on the edge of the parity gate (r34 failed on it).
if [ "${SERVER_WEIGHT_MODE:-lora}" = "flash" ]; then
  # No default: an engine with fewer slots than its share of a window queues
  # the episodes and collection slows silently. flash_rl.sbatch derives it from
  # the run's config (training.job_shape: the share plus 8).
  case "${SERVER_MAX_NUM_SEQS:-}" in
    "" | *[!0-9]* | 0*)
      echo "launch_platypus_server: flash requires SERVER_MAX_NUM_SEQS, a positive integer (got '${SERVER_MAX_NUM_SEQS:-}')" >&2
      exit 1
      ;;
  esac
  export VLLM_USE_V2_MODEL_RUNNER=1
  ARGS+=(
    --kv-cache-dtype "${SERVER_KV_CACHE_DTYPE:-auto}"
    --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode align
    --kernel-config '{"moe_backend":"marlin"}'
    --no-enable-flashinfer-autotune
    # Inert at fork 3bda1ba09, which serves Qwen4Exp without torch.compile
    # (breakable CUDA graphs); kept for a pin that compiles again, where
    # compile-time tuning cloned the host PLE table onto the GPU (OOM).
    --compilation-config '{"inductor_compile_config":{"triton.autotune_at_compile_time":false}}'
    --enable-chunked-prefill
    --max-num-seqs "${SERVER_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${SERVER_MAX_NUM_BATCHED_TOKENS:-8192}"
    --gpu-memory-utilization "${SERVER_GPU_MEMORY_UTILIZATION:-0.95}"
    # The harness deployment's scheduling (ullm deployment.build()): the next
    # step scheduled while the current one runs, stated so an incompatible
    # setting fails at startup rather than being dropped with a warning (the
    # fork enables it when unset); and 10% of the KV pool kept free when a
    # waiting request is admitted beside running ones, so a full pool queues
    # admissions instead of preempting decodes. Neither changes a draw.
    --async-scheduling --watermark 0.10
    --mm-processor-cache-gb 0 --skip-mm-profiling
    # Only the frozen PLE n-gram table lives on the host, born in pinned
    # memory and read over UVA (the fork's Engram; the harness deployment's
    # setting); this is NOT KV. The fork's generic UVA offload of the same
    # table (--cpu-offload-params) would first build it on the GPU.
    # Keep the pass-through CPU/KV guard above: no generic offload overrides.
    --engram-config '{"cpu_offload": true}'
  )
  # MTP drafting as ullm's e307 deployment drafts (its speculative_config and
  # FLASHNEXT_E307_DRAFTER, from inference.speculative_decoding): sampled
  # drafts, exact rejection sampling, GDN RecoverSSM. Off unless set.
  if [ -n "${SERVER_SPECULATIVE_CONFIG:-}" ]; then
    ARGS+=(--speculative-config "${SERVER_SPECULATIVE_CONFIG}")
  fi
else
  ARGS+=(--additional-config "{\"gdn_prefill_backend\": \"${SERVER_GDN_PREFILL_BACKEND:-triton}\"}")
fi
# Attention backend override (FLASHINFER | FLASH_ATTN | TRITON_ATTN ...) and a
# raw pass-through for parity bisections; both default to vLLM's choice.
if [ -n "${SERVER_ATTENTION_BACKEND:-}" ]; then
  ARGS+=(--attention-backend "${SERVER_ATTENTION_BACKEND}")
fi
if [ -n "${SERVER_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${SERVER_EXTRA_ARGS})
fi

# vLLM's engine JIT (flashinfer/deep_gemm) needs a discoverable CUDA
# toolkit; the cluster ships driver-only nodes, and ullm's set_env stitches
# the pip nvidia-cuda-nvcc wheel into a usable CUDA_HOME. Importing ullm
# into the server process is NOT an option (its nest_asyncio patch breaks
# vllm's uvloop.run), so a throwaway interpreter computes the env delta and
# the clean vllm binary inherits it.
ENV_DELTA="$("${PLATYPUS_VENV}/bin/python" -c '
import json, os
before = dict(os.environ)
from ullm.backends import config_local
config_local.set_env()
delta = {k: v for k, v in os.environ.items() if before.get(k) != v}
print(json.dumps(delta))
')"
while IFS="=" read -r key value; do
  export "$key=$value"
done < <(printf "%s" "$ENV_DELTA" | "${PLATYPUS_VENV}/bin/python" -c '
import json, sys
for key, value in json.load(sys.stdin).items():
    print(f"{key}={value}")
')

# Durable node-local compile caches: reuse vLLM/inductor/triton compile
# artifacts across runs on the node instead of ullm's /tmp default or NFS
# home. Applied AFTER the
# ullm env delta so these win.
# No fallback: a compile cache that silently lands in NFS ~/.cache is exactly
# what this exists to avoid, so the job must hand us the node-local root.
: "${ARC3_RL_CACHE_ROOT:?set ARC3_RL_CACHE_ROOT (the node-local shared cache root)}"
ARC3_COMPILE_CACHE_ROOT="${ARC3_COMPILE_CACHE_ROOT:-${ARC3_RL_CACHE_ROOT}/compile}"
if mkdir -p "${ARC3_COMPILE_CACHE_ROOT}" 2>/dev/null; then
  export VLLM_CACHE_ROOT="${ARC3_COMPILE_CACHE_ROOT}/vllm-platypus"
  export TORCHINDUCTOR_CACHE_DIR="${ARC3_COMPILE_CACHE_ROOT}/inductor-platypus"
  export TRITON_CACHE_DIR="${ARC3_COMPILE_CACHE_ROOT}/triton-platypus"
  export XDG_CACHE_HOME="${ARC3_COMPILE_CACHE_ROOT}/xdg-platypus"
  # FlashInfer 0.6.13 ignores XDG and appends .cache/flashinfer to this base.
  # Its build.ninja embeds absolute package paths: share only within one venv.
  __arc3_flashinfer_env_key="$(printf '%s' "${PLATYPUS_VENV}" | sha256sum | cut -c1-16)"
  export FLASHINFER_WORKSPACE_BASE="${ARC3_COMPILE_CACHE_ROOT}/flashinfer-${__arc3_flashinfer_env_key}"
  mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer"
  echo "launch_platypus_server: compile caches under ${ARC3_COMPILE_CACHE_ROOT}" >&2
  echo "launch_platypus_server: FlashInfer cache ${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer" >&2
else
  echo "launch_platypus_server: FATAL ${ARC3_COMPILE_CACHE_ROOT} not writable" >&2
  exit 1
fi

# Make the native service importable even when invoked from
# another working directory or with a copied node-local environment.
ARC3_SERVER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ARC3_SERVER_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Invoke the native service through the venv's Python: the console
# script's shebang is the ORIGINAL venv path, so a node-local copy of the
# venv (platypus_debug_smoke.sbatch syncs one off NFS) would otherwise import
# from the slow NFS site-packages it was copied from.
if [ -n "${SERVER_LOG:-}" ]; then
  exec "${PLATYPUS_VENV}/bin/python" -m training.native_vllm_server "${ARGS[@]}" >> "${SERVER_LOG}" 2>&1
else
  exec "${PLATYPUS_VENV}/bin/python" -m training.native_vllm_server "${ARGS[@]}"
fi
