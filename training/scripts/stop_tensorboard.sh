#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --port <1..65535>" >&2
}

if [[ "$#" -ne 2 || "$1" != "--port" ]]; then
  usage
  exit 2
fi
port="$2"
if [[ ! "${port}" =~ ^[0-9]+$ || "${port}" -lt 1 || "${port}" -gt 65535 ]]; then
  echo "invalid TensorBoard port: ${port}" >&2
  usage
  exit 2
fi

# Never select this script, its shell, or make/python ancestors even if their
# command line happens to quote the TensorBoard command being launched next.
declare -A keep_pids=()
pid="$$"
while [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 ]]; do
  keep_pids["${pid}"]=1
  pid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
done

declare -A parent_by_pid=()
declare -A args_by_pid=()
while read -r process_pid parent_pid process_args; do
  [[ -n "${process_pid:-}" && -n "${parent_pid:-}" && -n "${process_args:-}" ]] || continue
  parent_by_pid["${process_pid}"]="${parent_pid}"
  args_by_pid["${process_pid}"]="${process_args}"
done < <(ps -u "$(id -u)" -o pid=,ppid=,args=)

declare -A selected=()
for process_pid in "${!args_by_pid[@]}"; do
  [[ -z "${keep_pids[${process_pid}]:-}" ]] || continue
  process_args="${args_by_pid[${process_pid}]}"
  # Match an actual TensorBoard argv and this exact port. The old script also
  # matched every tensorboard_data_server owned by the user, killing unrelated
  # repos and ports.
  if [[ "${process_args}" =~ (^|[[:space:]/])tensorboard([[:space:]]|$) ]] \
      && [[ "${process_args}" =~ (^|[[:space:]])--port(=|[[:space:]])${port}([[:space:]]|$) ]]; then
    selected["${process_pid}"]=1
  fi
done

# Data-server children do not carry --port. Capture descendants of the scoped
# TensorBoard root before signalling it so they are cleaned up without any
# global name match.
changed=1
while [[ "${changed}" -eq 1 ]]; do
  changed=0
  for process_pid in "${!parent_by_pid[@]}"; do
    [[ -z "${keep_pids[${process_pid}]:-}" ]] || continue
    [[ -z "${selected[${process_pid}]:-}" ]] || continue
    if [[ -n "${selected[${parent_by_pid[${process_pid}]}]:-}" ]]; then
      selected["${process_pid}"]=1
      changed=1
    fi
  done
done

matching_pids=("${!selected[@]}")
if [[ "${#matching_pids[@]}" -eq 0 ]]; then
  echo "No TensorBoard process found on port ${port}"
  exit 0
fi

kill "${matching_pids[@]}" 2>/dev/null || true
sleep 1

remaining_pids=()
for process_pid in "${matching_pids[@]}"; do
  if kill -0 "${process_pid}" 2>/dev/null; then
    remaining_pids+=("${process_pid}")
  fi
done
if [[ "${#remaining_pids[@]}" -gt 0 ]]; then
  kill -9 "${remaining_pids[@]}" 2>/dev/null || true
fi

printf 'Stopped TensorBoard processes on port %s: %s\n' "${port}" "${matching_pids[*]}"
