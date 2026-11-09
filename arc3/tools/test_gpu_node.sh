#!/bin/bash
# Node-side wrapper for `make test-gpu` (invoked by `make test-gpu-srun`
# inside the staged copy on shared storage). Mirrors taaf.deploy_slurm's
# node-local-storage resolution: --export=ALL leaks the login node's
# cache/tmp env, which points at paths that don't exist on compute nodes,
# so clear it and resolve genuinely node-local roots; build the venv there
# (an NFS venv with the vllm stack is metadata-bound and crawls).
set -euo pipefail

[ -f /etc/kander/job-env.sh ] && . /etc/kander/job-env.sh
unset UV_CACHE_DIR TMPDIR TEMPDIR UV_PYTHON_INSTALL_DIR MPLCONFIGDIR XDG_CACHE_HOME

user="${USER:-$(id -un)}"
uid="$(id -u)"
job="${SLURM_JOB_ID:-$$}"

if [ -n "${SCRATCH_DIR:-}" ] && [ -d "$SCRATCH_DIR" ] && [ -w "$SCRATCH_DIR" ]; then
    root="$SCRATCH_DIR"
elif [ -n "${SLURM_TMPDIR:-}" ] && [ -d "$SLURM_TMPDIR" ] && [ -w "$SLURM_TMPDIR" ]; then
    root="$SLURM_TMPDIR"
elif [ -n "${SLURM_JOB_ID:-}" ] && mkdir -p "/scratch/jobs/$job" 2>/dev/null && [ -w "/scratch/jobs/$job" ]; then
    root="/scratch/jobs/$job"
else
    root="/tmp/$user/platypus-gpu-test"
    mkdir -p "$root"
fi
# Persistent per-node cache where the cluster provides one, else job-local.
if mkdir -p "/cache/users/$uid/platypus" 2>/dev/null && [ -w "/cache/users/$uid/platypus" ]; then
    cache="/cache/users/$uid/platypus"
else
    cache="$root/cache"
    mkdir -p "$cache"
fi

export UV_CACHE_DIR="$cache/uv"
export UV_PYTHON_INSTALL_DIR="$cache/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$cache/xdg"
export TMPDIR="$root/tmp"
# flashinfer JIT-caches ninja files with ABSOLUTE paths into the venv; both
# must therefore be stable across jobs (and off shared $HOME, where one
# node's cache would poison another's). The persistent per-node cache root
# hosts both, so JIT artifacts stay valid job after job.
export FLASHINFER_WORKSPACE_BASE="$cache"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
# HF model cache stays at its $HOME default: shared, so models download once.

repo="$(dirname "$(readlink -f "$0")")/.."
# Venv keyed by the staged checkout's name: parallel forks run their gates
# concurrently with different uv.locks, and a shared venv ping-pongs
# between them mid-sync.
venv="$cache/venv-$(basename "$(readlink -f "$repo")")"
echo "[test_gpu_node] node=$(hostname) root=$root cache=$cache venv=$venv args=$*"
# Parallel column jobs can land on the same node and share the venv:
# serialize the sync under a lock; the post-lock sync in a waiter is a
# fast no-op. Extra script args (e.g. `-k direct`) pass through to pytest.
# A job killed mid-write (the srun time limit) can leave the PERSISTENT node
# caches corrupt (Python "missing _sysconfigdata_", uv archive missing
# METADATA), poisoning every later run on the node. Heal entirely UNDER
# the lock: first plain retry (another column may have healed already —
# wiping here would delete a venv its pytest is running from), then wipe
# the cached Python installs, the uv cache, and this venv, and rebuild.
if ! flock "$cache/install.lock" make -C "$repo" install-gpu VENV="$venv"; then
    echo "[test_gpu_node] install failed — healing under lock (retry, then wipe caches + retry)"
    flock "$cache/install.lock" sh -c "
        make -C \"$repo\" install-gpu VENV=\"$venv\" && exit 0
        echo '[test_gpu_node] retry failed too — wiping cached python/uv-cache/venv'
        rm -rf \"$UV_PYTHON_INSTALL_DIR\" \"$UV_CACHE_DIR\" \"$venv\"
        make -C \"$repo\" install-gpu VENV=\"$venv\"
    "
fi
# A successful install does not mean a working interpreter, so the same
# corruption has a second shape the block above cannot see: `uv sync` is
# satisfied by an install DIRECTORY, and a wipe interrupted partway leaves
# the cached Python present but without its stdlib. uv skips re-downloading
# it, reports success, and every job on that node then dies at startup with
# "No module named 'encodings'" — silently, run after run, since nothing
# ever asks the installer again. Ask the interpreter instead.
if ! "$venv/bin/python" -V >/dev/null 2>&1; then
    echo "[test_gpu_node] venv python does not start — healing under lock"
    flock "$cache/install.lock" sh -c "
        \"$venv/bin/python\" -V >/dev/null 2>&1 && exit 0
        echo '[test_gpu_node] wiping cached python/uv-cache/venv'
        rm -rf \"$UV_PYTHON_INSTALL_DIR\" \"$UV_CACHE_DIR\" \"$venv\"
        make -C \"$repo\" install-gpu VENV=\"$venv\"
    "
fi
exec make -C "$repo" test-gpu-only VENV="$venv" TEST_ARGS="$*"
