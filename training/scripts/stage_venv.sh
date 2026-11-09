# shellcheck shell=bash
# Stage a packed venv (built by scripts/build_env.sbatch, packed by scripts/pack_venv.sh; see
# docs/environments.md) into node-local /cache and export its interpreter. Source this from a
# Slurm batch script, after job-env.sh:
#
#   . /etc/kander/job-env.sh
#   . scripts/stage_venv.sh
#   stage_venv /data/personal/$USER/arc-training-envs/trainer-flash-distillation-<key>.tar.zst <key> ARC3_FLASH_PYTHON
#
# Layout and rules (Kander storage guidance for reusable environments):
# - one directory per key under the per-user cache, /cache/users/$UID/venvs/<name>-<key>
#   (STAGE_VENV_ROOT overrides the root; it must stay under /cache/users/$UID/);
# - private files (umask 077): this is a per-user cache, not /cache/shared;
# - a cold stage runs under flock on <root>/.<name>-<key>.lock: one sequential
#   read of the tarball from /data, sha256 check against the manifest, extraction
#   into a temp dir, rewrite of the recorded build-path references, an atomic
#   rename into place, an interpreter check, and only then the .venv-ready marker;
# - a staged directory is never modified or replaced in place; a failed stage
#   only ever removes its own .tmp-* directories (path-guarded);
# - warm path (marker present, nothing evicted, interpreter runs) takes no lock.
# /cache is evictable: a directory that lost files is moved aside and restaged.
# The locked part runs as a separate bash process (this file with __locked) so
# errexit and its cleanup traps hold even when the caller uses `|| ...`.

_STAGE_VENV_SELF=$(readlink -f "${BASH_SOURCE[0]}")

_stage_venv_log() { echo "stage_venv: $*" >&2; }

# Remove one of our temp directories; refuses anything else.
_stage_venv_rm() {
  local root=$1 path=$2
  [[ -e "$path" || -L "$path" ]] || return 0
  if [[ "$root" != /cache/users/*/* || "$path" != "$root"/.tmp-* ]]; then
    _stage_venv_log "refusing to remove $path"
    return 1
  fi
  rm -rf -- "$path"
}

# Entries outside __pycache__ and the marker (bytecode written after staging does not count).
_stage_venv_count() {
  find "$1" \( -path '*/__pycache__' -o -name '.venv-ready*' \) -prune -o -print | wc -l
}

_stage_venv_ready() {
  local dest=$1 recorded current
  [[ -f "$dest/.venv-ready" ]] || return 1
  recorded=$(sed -n 's/^files=//p' "$dest/.venv-ready")
  current=$(_stage_venv_count "$dest")
  [[ -n "$recorded" && "$current" -ge "$recorded" ]] || return 1
  timeout 120 "$dest/bin/python" -c 'import sys' >/dev/null 2>&1
}

# Executed (not sourced) with errexit: _stage_venv_locked TARBALL MANIFEST ROOT DEST NAME KEY
_stage_venv_locked() {
  umask 077
  tarball=$1 manifest=$2 root=$3 dest=$4 name=$5 key=$6
  start=$SECONDS child="" tmp=""
  if _stage_venv_ready "$dest"; then
    _stage_venv_log "$name-$key warm at $dest ($((SECONDS - start)) s)"
    return 0
  fi
  mkdir -p "$root"
  exec {lockfd}>"$root/.$name-$key.lock"
  if ! flock -w "${STAGE_VENV_LOCK_WAIT:-3600}" "$lockfd"; then
    _stage_venv_log "timed out waiting for $root/.$name-$key.lock"
    return 1
  fi
  if _stage_venv_ready "$dest"; then
    _stage_venv_log "$name-$key staged by another job while waiting ($((SECONDS - start)) s)"
    return 0
  fi
  # Holding the key's lock: earlier temp dirs of this key belong to dead stages.
  for leftover in "$root/.tmp-$name-$key".*; do _stage_venv_rm "$root" "$leftover"; done
  if [[ -e "$dest" ]]; then
    _stage_venv_log "$dest is incomplete (evicted or unverified); moving it aside and restaging"
    mv -T "$dest" "$root/.tmp-$name-$key.stale-$$"
    _stage_venv_rm "$root" "$root/.tmp-$name-$key.stale-$$"
  fi
  tmp=$(mktemp -d "$root/.tmp-$name-$key.XXXXXX")
  trap 'if [[ -n "${child:-}" ]]; then kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; fi; _stage_venv_rm "$root" "${tmp:-}"' EXIT
  trap 'exit 143' TERM INT

  expected=$(sed -n 's/^ *"tarball_sha256": "\([0-9a-f]\{64\}\)".*/\1/p' "$manifest")
  unpacked=$(sed -n 's/^ *"unpacked_bytes": \([0-9]*\).*/\1/p' "$manifest")
  if [[ ${#expected} -ne 64 || -z "$unpacked" ]]; then
    _stage_venv_log "manifest $manifest lacks tarball_sha256/unpacked_bytes"
    return 1
  fi
  tarsize=$(stat -c %s "$tarball")
  avail=$(df -P -B1 "$root" | awk 'NR == 2 {print $4}')
  if (( avail < tarsize + unpacked + 10 * 1024 ** 3 )); then
    _stage_venv_log "not enough space in $root: need $((tarsize + unpacked)) B + 10 GiB, have $avail B"
    return 1
  fi

  t0=$SECONDS
  timeout 30m cp "$tarball" "$tmp/venv.tar.zst" & child=$!   # one sequential read from /data
  wait "$child"; child=""
  copy_s=$((SECONDS - t0)); t0=$SECONDS
  actual=$(sha256sum "$tmp/venv.tar.zst" | cut -c1-64)
  if [[ "$actual" != "$expected" ]]; then
    _stage_venv_log "sha256 mismatch for $tarball: $actual != $expected"
    return 1
  fi
  sha_s=$((SECONDS - t0)); t0=$SECONDS
  mkdir "$tmp/venv"
  timeout 30m tar -I zstd -xf "$tmp/venv.tar.zst" -C "$tmp/venv" & child=$!
  wait "$child"; child=""
  rm -f "$tmp/venv.tar.zst"
  extract_s=$((SECONDS - t0)); t0=$SECONDS
  if [[ ! -x "$tmp/venv/.python/bin/python3" || ! -f "$tmp/venv/.venv-pack/original-prefix" ]]; then
    _stage_venv_log "$tarball is not a pack_venv.sh artifact"
    return 1
  fi
  # Point the recorded build-path references (shebangs, activate scripts,
  # pyvenv.cfg home) at the final location; the bundled base interpreter does it.
  timeout 10m "$tmp/venv/.python/bin/python3" - "$tmp/venv" "$dest" <<'PY'
import pathlib, sys
root, new = pathlib.Path(sys.argv[1]), sys.argv[2].encode()
old = (root / ".venv-pack/original-prefix").read_text().strip().encode()
listed = [rel for rel in (root / ".venv-pack/prefix-files").read_bytes().split(b"\0") if rel]
for rel in listed:
    path = root / rel.decode()
    path.write_bytes(path.read_bytes().replace(old, new))
    if old in path.read_bytes():
        raise SystemExit(f"{path} still references {old.decode()}")
print(f"stage_venv: rewrote {len(listed)} build-path references", file=sys.stderr)
PY
  mv -T "$tmp/venv" "$dest"
  if ! timeout 120 "$dest/bin/python" -c '
import os, sys
assert os.path.realpath(sys.prefix) == os.path.realpath(sys.argv[1]), (sys.prefix, sys.argv[1])
assert os.path.realpath(sys.base_prefix) == os.path.realpath(os.path.join(sys.argv[1], ".python")), sys.base_prefix
import encodings, json, site, ssl  # noqa: F401
' "$dest"; then
    _stage_venv_log "interpreter check failed in $dest; removing it"
    mv -T "$dest" "$tmp/failed"
    return 1
  fi
  files=$(_stage_venv_count "$dest")
  printf 'tarball=%s\nsha256=%s\nfiles=%s\nstaged_at=%s\nslurm_job=%s\n' \
    "$tarball" "$expected" "$files" "$(date -Is)" "${SLURM_JOB_ID:-none}" > "$dest/.venv-ready.tmp"
  mv -T "$dest/.venv-ready.tmp" "$dest/.venv-ready"
  _stage_venv_log "$name-$key cold stage to $dest: copy ${copy_s}s ($((tarsize / 1048576)) MiB), sha256 ${sha_s}s, extract ${extract_s}s, rewrite+check $((SECONDS - t0))s, total $((SECONDS - start))s"
}

# stage_venv TARBALL KEY VAR: stage TARBALL (named <name>-<KEY>.tar.zst, with
# <name>-<KEY>.json beside it) and export VAR=<staged venv>/bin/python.
stage_venv() {
  local tarball=${1:?stage_venv TARBALL KEY VAR} key=${2:?stage_venv TARBALL KEY VAR} var=${3:?stage_venv TARBALL KEY VAR}
  local uid root base name manifest
  uid=$(id -u)
  root=${STAGE_VENV_ROOT:-/cache/users/$uid/venvs}
  base=${tarball##*/}
  base=${base%.tar.zst}
  name=${base%-"$key"}
  manifest=${tarball%.tar.zst}.json
  if [[ ! "$key" =~ ^[0-9a-f]{16}$ || "$base" != "$name-$key" || -z "$name" || "$tarball" != /* ]]; then
    _stage_venv_log "expected an absolute <name>-<16-hex key>.tar.zst path and its key, got $tarball $key"
    return 1
  fi
  if [[ ! "$var" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
    _stage_venv_log "bad variable name $var"
    return 1
  fi
  if [[ "$root" != /cache/users/"$uid"/* ]]; then
    _stage_venv_log "STAGE_VENV_ROOT must be under /cache/users/$uid/, got $root"
    return 1
  fi
  if [[ ! -f "$tarball" || ! -f "$manifest" ]]; then
    _stage_venv_log "missing $tarball or its manifest $manifest"
    return 1
  fi
  bash "$_STAGE_VENV_SELF" __locked "$tarball" "$manifest" "$root" "$root/$name-$key" "$name" "$key" || return 1
  export "$var=$root/$name-$key/bin/python"
  _stage_venv_log "$var=$root/$name-$key/bin/python"
}

# stage_venv_print TARBALL VAR: stage_venv with the key taken from TARBALL's
# name, then print the interpreter path (for callers that capture stdout).
stage_venv_print() {
  local tarball=${1:?stage_venv_print TARBALL VAR} var=${2:?stage_venv_print TARBALL VAR} base
  base=${tarball##*/}
  base=${base%.tar.zst}
  stage_venv "$tarball" "${base##*-}" "$var" || return 1
  printf '%s\n' "${!var}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  if [[ "${1:-}" != __locked || $# -ne 7 ]]; then
    echo "source this file and call stage_venv TARBALL KEY VAR (see the header)" >&2
    exit 2
  fi
  shift
  _stage_venv_locked "$@"
fi
