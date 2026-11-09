#!/bin/bash
# Execute only the package snapshotted by drafter.jobs.
set -euo pipefail
job_dir="${1:?job directory}"
runtime_python="${2:-}"
source_dir="${job_dir}/source"
cache_root="${DRAFTER_CACHE_ROOT:-/cache/users/$(id -u)/drafter}"
mkdir -p "${cache_root}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"
ulimit -Sn "$(ulimit -Hn)"
unset VIRTUAL_ENV
if [ -n "${runtime_python}" ]; then
  if [ ! -x "${runtime_python}" ]; then
    echo "Serving interpreter is not executable on this node: ${runtime_python}" >&2
    exit 1
  fi
  mkdir -p "${cache_root}/venvs"
  export UV_PROJECT_ENVIRONMENT
  UV_PROJECT_ENVIRONMENT="$(mktemp -d "${cache_root}/venvs/serving-XXXXXXXX")"
  trap 'rm -rf -- "${UV_PROJECT_ENVIRONMENT}"' EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  # Install locally over a qualified serving stack. Never modify that stack.
  uv venv --python "${runtime_python}" --system-site-packages "${UV_PROJECT_ENVIRONMENT}"
  # A venv created from another venv otherwise sees only the base Python's
  # system packages. Reuse the qualified interpreter's site directories,
  # including their editable-install .pth files, behind our local package.
  "${runtime_python}" - "${UV_PROJECT_ENVIRONMENT}" <<'PYENV'
import json
import site
import sys
from pathlib import Path
venv = Path(sys.argv[1])
paths = site.getsitepackages()
target = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
(target / "serving_runtime.pth").write_text(
    "import site; " + "; ".join(f"site.addsitedir({path!r})" for path in paths) + "\n"
)
(venv / "serving_runtime.json").write_text(json.dumps({"python": sys.executable, "sites": paths}, indent=2))
PYENV
  uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --no-deps "${source_dir}"
else
  source "${source_dir}/tools/environment.sh"
  setup_drafter_environment "${source_dir}" "${cache_root}"
fi
export PYTHONPATH="${source_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${job_dir}"
"${UV_PROJECT_ENVIRONMENT}/bin/python" run_in_worker.py
