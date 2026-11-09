#!/bin/bash
# Reuse dependencies only. The caller selects its own package via PYTHONPATH.
setup_drafter_environment() {
  local source_dir="${1:?package source}" cache_root="${2:?cache root}" key
  key="$(cat "${source_dir}/uv.lock" "${source_dir}/pyproject.toml" | sha256sum | cut -c1-64)"
  export UV_PROJECT_ENVIRONMENT="${cache_root}/venvs/locked-${key}"
  mkdir -p "${cache_root}/venvs"
  (
    flock -w 3600 9
    if [ ! -f "${UV_PROJECT_ENVIRONMENT}/.ready" ]; then
      uv sync --project "${source_dir}" --locked --no-dev --no-install-project
      touch "${UV_PROJECT_ENVIRONMENT}/.ready"
    fi
  ) 9>"${UV_PROJECT_ENVIRONMENT}.lock"
}
