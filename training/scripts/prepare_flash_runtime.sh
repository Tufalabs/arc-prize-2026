#!/bin/bash
# Materialize the isolated Flash-Next runtime on the node-local cache and print
# its interpreter. The trainer venv (vLLM 0.25.1, Transformers 5.14) cannot
# load Qwen3.8-Flash-Next; this runtime is the attested archive the router
# experiments ran on (vLLM 0.29 fork 29da4c8, Transformers 5.16.1, torch 2.13),
# plus the verified Python overlay bundle, plus the trainer packages the archive
# lacks (peft, accelerate, tensorboard). Never installs into a shared venv.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ARC3_RL_CACHE_ROOT:?set ARC3_RL_CACHE_ROOT}"
archive="${ARC3_FLASH_RUNTIME_ARCHIVE:-/home/$USER/training_envs/platypus_main_977964_vllm29da4/runtime.tar}"
bundle="${ARC3_FLASH_OVERLAY:-/data/personal/$USER/flash_runtime/runtime_overlay.tar.gz}"
expected_base="26c5fd32c9b4fab7c055fe3895d39119a97bea5b96a531bbfcc02b0e588a11c1"
expected_bundle="1da49e560a707bbbf7a62fd9c1bfb0369d4712b512c9816aee6e7c5803396a9b"
# Pinned to uv.lock's versions so the trainer code sees one peft everywhere.
extras="peft==0.18.1 accelerate==1.13.0 tensorboard==2.20.0"
recipe="flash-runtime-v2"
if [ ! -f "${archive}" ] || [ ! -f "${bundle}" ]; then
  echo "Missing Flash archive ${archive} or overlay bundle ${bundle}" >&2
  exit 1
fi
bundle_sha="$(sha256sum "${bundle}" | cut -d' ' -f1)"
if [ "${bundle_sha}" != "${expected_bundle}" ]; then
  echo "Overlay bundle checksum ${bundle_sha} does not match the pinned ${expected_bundle}" >&2
  exit 1
fi
key="$(printf '%s\n%s\n%s\n' "${bundle_sha}" "${extras}" "${recipe}" | sha256sum | cut -c1-20)"
runtime_parent="${ARC3_RL_CACHE_ROOT}/flash/runtimes"
runtime="${runtime_parent}/${key}"
mkdir -p "${runtime_parent}"
exec 9>"${runtime}.lock"
flock -w 3600 9
if [ ! -f "${runtime}/.flash-ready" ]; then
  if [ -e "${runtime}" ]; then
    # Only a build that died under this lock leaves a directory without the
    # ready marker; nothing else ever writes here. Rebuild from scratch.
    echo "Removing incomplete runtime at ${runtime}" >&2
    rm -rf "${runtime}"
  fi
  echo "Verifying and unpacking the Flash runtime into ${runtime}" >&2
  actual_base="$(sha256sum "${archive}" | cut -d' ' -f1)"
  if [ "${actual_base}" != "${expected_base}" ]; then
    echo "Flash archive checksum does not match the pinned base" >&2
    exit 1
  fi
  mkdir "${runtime}"
  tar -xf "${archive}" -C "${runtime}"
  "${runtime}/bin/python" "${REPO_ROOT}/scripts/flash_runtime.py" apply \
    --runtime "${runtime}" --bundle "${bundle}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/cache/users/$(id -u)/uv}"
  mkdir -p "${UV_CACHE_DIR}"
  env -u VIRTUAL_ENV uv pip install \
    --python "${runtime}/bin/python" --no-deps --no-sources --no-build-isolation \
    "${runtime}/source/platypus/packages/platypus" \
    "${runtime}/source/platypus/packages/duck" \
    "${runtime}/source/platypus/packages/ullm" \
    "${runtime}/source/framework" >&2
  # shellcheck disable=SC2086
  env -u VIRTUAL_ENV uv pip install --python "${runtime}/bin/python" ${extras} >&2
  # The extras must not have moved the attested stack.
  # Everything but the final interpreter path goes to stderr: callers capture stdout.
  "${runtime}/bin/python" - >&2 <<'PY'
import importlib.metadata as m
expected = {"vllm": "0.29.1.dev30+g29da4c899.precompiled", "transformers": "5.16.1", "torch": "2.13.0", "peft": "0.18.1"}
for name, version in expected.items():
    found = m.version(name)
    if found != version:
        raise SystemExit(f"{name} is {found}, expected {version}: the extras changed the attested runtime")
import peft, accelerate, tensorboard  # noqa: F401
print("flash runtime extras verified", flush=True)
PY
  touch "${runtime}/.flash-ready"
fi
printf '%s\n' "${runtime}/bin/python"
