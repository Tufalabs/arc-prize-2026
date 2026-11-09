#!/usr/bin/env bash
# Run the gpu tier on compute nodes: one test per job, all at once.
#
# Two reasons it is one test per job rather than one job for the tier. Each
# of these boots its own vLLM engine, and stacking a second into a process
# that has already built one wedges engine startup — it sits there until the
# walltime rather than failing. And a warm node cache syncs in seconds while
# the tests are minutes each, so the tier's wall clock is the sum of the
# tests unless they run at once.
#
# The checkout lives on the login node's own /scratch, which a compute node
# cannot see, so the repo is staged to shared storage and every job runs
# from there.
#
# With no arguments this is the login side: collect, stage, fan out, wait.
# With a pytest node id — which is how it re-enters itself under srun — it is
# the node side: build a venv in node-local space and run that one test.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Which slice of the tier this invocation is for, and what to call it. The
# duck's engine test needs a card an order of magnitude bigger than the seam
# checks do, so the tier is dispatched in two shapes rather than one; the
# label keeps their staging directories and logs apart, which is what lets
# the two run at the same time.
# Exported, not passed: srun carries the caller's environment to the node
# (--export=ALL is its default), and an `env` prefix on the command line
# resolves against PATH on the node — where a user-local `env` shim that is
# not executable there turns the whole column into exit 13 before Python
# starts. That is how the first B200 column died.
export ARC2_GPU_MARK="${ARC2_GPU_MARK:-gpu}"
export ARC2_GPU_LABEL="${ARC2_GPU_LABEL:-gpu}"
mark_expr="${ARC2_GPU_MARK}"
label="${ARC2_GPU_LABEL}"
stage_parent="${ARC2_GPU_STAGE:-/data/personal/${USER}/arc2-gpu-stage-${label}}"
stage_root="${stage_parent}/arc2"
log_dir="${repo_root}/.gpu-test-logs/${label}"

if [ "$#" -eq 0 ]; then
    # --- login side ---------------------------------------------------------
    # Collected, not listed: a hardcoded set of columns is one that silently
    # stops covering a test someone adds. Only pytest's own "nothing
    # matched" (exit 5) is the empty tier; a collection error must not read
    # as one, or an import failure becomes a green tier.
    collect_rc=0
    collected="$(uv run pytest -m "${mark_expr}" --collect-only -q 2>&1)" || collect_rc=$?
    if [ "${collect_rc}" -eq 5 ]; then
        echo "  (${label}: no test matches -m '${mark_expr}' yet)"
        exit 0
    elif [ "${collect_rc}" -ne 0 ]; then
        echo "${collected}"
        echo "collecting -m '${mark_expr}' failed (pytest exit ${collect_rc})" >&2
        exit "${collect_rc}"
    fi
    mapfile -t gpu_tests < <(printf '%s\n' "${collected}" | grep '::')
    echo "staging ${repo_root} -> ${stage_root}"
    mkdir -p "${stage_root}" "${log_dir}"
    rsync -a --delete \
        --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
        --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude '.gpu-test-logs' \
        --exclude 'notebooks/runs' --exclude '.data-src' \
        "${repo_root}/" "${stage_root}/"
    # The LLM layer comes from the ARC3 project beside this one.
    rsync -a --delete --exclude '.venv' --exclude '__pycache__' --exclude '.gpu-test-logs' \
        "${repo_root}/../arc3/" "${stage_parent}/arc3/"

    pids=()
    for test_id in "${gpu_tests[@]}"; do
        column="${test_id##*::}"
        echo "[${label}] launching ${column}"
        srun \
            --account="${ARC2_GPU_ACCOUNT:-research}" \
            --partition="${ARC2_GPU_PARTITION:-dev}" \
            --ntasks=1 \
            --gres="gpu:${ARC2_GPU_TYPE:-b200}:1" \
            --cpus-per-gpu="${ARC2_GPU_CPUS:-8}" \
            --mem-per-gpu="${ARC2_GPU_MEM:-96G}" \
            --time="${ARC2_GPU_TIME:-01:00:00}" \
            --job-name="arc2-${label}-${column}" \
            bash "${stage_root}/tools/test_gpu_node.sh" "${test_id}" \
            > "${log_dir}/${column}.log" 2>&1 &
        pids+=("$!:${column}")
    done

    status=0
    for entry in "${pids[@]}"; do
        pid="${entry%%:*}"
        column="${entry##*:}"
        if wait "${pid}"; then
            echo "[${label}] column ${column}: OK"
        else
            echo "[${label}] column ${column}: FAILED (see ${log_dir}/${column}.log)"
            status=1
        fi
    done
    exit "${status}"
fi

# --- node side -------------------------------------------------------------
node_root="/scratch/jobs/${SLURM_JOB_ID:-$$}"
mkdir -p "${node_root}" 2>/dev/null || node_root="/tmp/${USER}-arc2-${SLURM_JOB_ID:-$$}"
cache_root="/cache/users/$(id -u)"
mkdir -p "${cache_root}" 2>/dev/null || cache_root="${node_root}/cache"

# Node-local for the venv, not shared: a uv sync over NFS is minutes of
# small-file traffic.
export UV_CACHE_DIR="${cache_root}/uv"
export UV_PYTHON_INSTALL_DIR="${cache_root}/python"
export UV_PROJECT_ENVIRONMENT="${node_root}/venv"
export UV_LINK_MODE=copy
export TMPDIR="${node_root}/tmp"
export XDG_CACHE_HOME="${cache_root}/xdg"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export HF_HOME="${cache_root}/hf"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${HF_HOME}"

cd "${stage_root}"
echo "$1 on $(hostname): venv=${UV_PROJECT_ENVIRONMENT} cache=${UV_CACHE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# One sync at a time per node. The columns land on whatever nodes are free,
# so two of them sharing one is ordinary, and concurrent syncs into a single
# cache are how a wheel ends up half-extracted — which then poisons every
# later job on that node, ours or anyone's (roadmap.md). The lock keeps our
# own jobs off each other; nothing here can protect against another user's.
#
# So the caches also HEAL, because the lock cannot undo damage already done:
# a job killed mid-write leaves a partial Python install or a uv archive
# without its metadata, and every later job on that node dies before pytest
# starts — silently, run after run, since nothing ever asks the installer to
# try again. Observed: a cpython install holding only `share/terminfo`,
# failing every sync for two days with a bare ENOENT.
#
# Heal entirely under the lock, and retry once before wiping: a sibling
# column may have healed it already, and wiping then would delete the caches
# its own sync is reading.
sync_all() { uv sync --locked --all-packages; }

exec {lock_fd}>"${UV_CACHE_DIR}.lock"
flock "${lock_fd}"
if ! sync_all; then
    echo "sync failed — healing the node cache (retry, then wipe and retry)" >&2
    if ! sync_all; then
        echo "retry failed too — wiping cached python and uv cache" >&2
        rm -rf "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" "${UV_PROJECT_ENVIRONMENT}"
        mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"
        sync_all
    fi
fi
# A successful sync does not mean a working interpreter: uv is satisfied by
# an install DIRECTORY, so a wipe interrupted partway leaves the cached
# Python present but without its stdlib, uv skips re-downloading it, and the
# job dies at startup with "No module named 'encodings'" instead. Ask the
# interpreter itself rather than the installer.
if ! "${UV_PROJECT_ENVIRONMENT}/bin/python" -V > /dev/null 2>&1; then
    echo "venv python does not start — wiping cached python and rebuilding" >&2
    rm -rf "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" "${UV_PROJECT_ENVIRONMENT}"
    mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"
    sync_all
fi
flock -u "${lock_fd}"

# `-s`: pytest discards captured output for a test that PASSES, both streams
# alike, and the tests in this tier report measurements — a mean score, a
# token count — that are the reason to run them at all. Without it a green
# column tells you it was green and nothing else. Everything the job prints
# is tee'd to .gpu-test-logs/${label}, one file per column, so the extra
# volume costs nothing that has to be read live.
exec uv run pytest -m "${mark_expr}" -v -s --durations=0 --durations-min=0.05 "$@"
