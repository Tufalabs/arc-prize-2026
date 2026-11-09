#!/bin/bash
# Pack a venv into one compressed, self-contained artifact on /data for
# scripts/stage_venv.sh. scripts/build_env.sbatch calls it; run it on a compute
# node (inside a Slurm job), never on the login node:
#
#   pack_venv.sh copy SRC DEST
#       Parallel tree copy (one worker per allocated CPU; keeps mtimes, modes
#       as masked by the umask, and symlinks; ACLs from NFS are not copied).
#       NFS file operations are latency-bound, so a single
#       stream is far slower than the link allows.
#   pack_venv.sh pack VENV NAME ORIGINAL_PREFIX OUT_DIR [EXTRA_JSON]
#       VENV is a node-local copy under /scratch and is modified in place:
#       - the key is sha256 over NAME, the recipe, the Python version, every
#         installed distribution's METADATA, and, for distributions not
#         installed from an index (a local directory or a VCS checkout, whose
#         version does not change with its code), the digests of the files it
#         installed into site-packages. Same lock and recipe -> same key;
#         different code -> different key;
#       - text files that embed ORIGINAL_PREFIX (the path the venv was built
#         at: shebangs, activate scripts) are listed for stage_venv.sh to
#         rewrite; binary matches are listed in the manifest (not rewritten);
#       - the base interpreter VENV links to is bundled as VENV/.python, and
#         bin/python and pyvenv.cfg point at it, so a staged venv needs nothing
#         from /data;
#       - OUT_DIR/NAME-KEY.tar.zst is written as one zstd stream (hashed while
#         written), then OUT_DIR/NAME-KEY.json, the manifest whose presence
#         marks the artifact as published. EXTRA_JSON (provenance) is stored
#         in the manifest under "source". Published artifacts are immutable:
#         an existing NAME-KEY is reported and left untouched, and concurrent
#         packs of one key publish once (OUT_DIR/.NAME-KEY.publishing). An
#         existing tarball is first checked against its manifest (size and
#         sha256); a damaged one fails the pack (exit 4) and stays in place.
#       Prints the tarball path on stdout; logs go to stderr.
# PACK_VENV_WORK_PREFIX (default /scratch/) and PACK_VENV_OUT_PREFIX (default
# /data/) are the locations VENV and OUT_DIR must be under; tests point them
# at a temporary directory.
set -euo pipefail
shopt -s inherit_errexit
export LC_ALL=C
recipe=venv-pack-v2
workers=${SLURM_CPUS_PER_TASK:-1}
work_prefix=${PACK_VENV_WORK_PREFIX:-/scratch/}
out_prefix=${PACK_VENV_OUT_PREFIX:-/data/}
log() { echo "pack_venv: $*" >&2; }

copy_tree() {
  local src=$1 dest=$2
  mkdir -p "$dest"
  (cd "$src" && find . -mindepth 1 -type d -printf '%P\0') |
    xargs -0 -r -n 200 -P "$workers" sh -c 'cd "$0" && exec mkdir -p -- "$@"' "$dest"
  (cd "$src" && find . -mindepth 1 ! -type d -printf '%P\0' |
    xargs -0 -r -n 200 -P "$workers" sh -c 'exec cp -P --preserve=timestamps --parents -t "$0" -- "$@"' "$dest")
}

# Key material: one line per distribution. RECORD lines inside site-packages
# (not the dist-info itself, scripts under ../, or bytecode, which embed the
# build path) identify the code of a distribution with a direct_url.json.
key_material() {
  local name=$1 version=$2 site=$3 dist
  printf '%s\n%s\n%s\n' "$name" "$recipe" "$version"
  for dist in "$site"/*.dist-info; do
    printf '%s %s\n' "${dist##*/}" "$(sha256sum < "$dist/METADATA" | cut -c1-64)"
    if [[ -f "$dist/direct_url.json" ]]; then
      printf '  files %s\n' "$({ grep -vE '^(\.\./|[^,]*\.dist-info/|[^,]*__pycache__/)' "$dist/RECORD" || true; } |
        sort | sha256sum | cut -c1-64)"
    fi
  done
}

# A published key's tarball must still be the bytes its manifest records: the
# build then checks a copy staged from it, possibly an older copy already on
# this node, so a damaged tarball would otherwise go unnoticed until a job
# stages it. Never repaired in place (artifacts are immutable).
verify_published() {
  local tarball=$1 manifest=$2 expected bytes size actual t0=$SECONDS
  expected=$(sed -n 's/^ *"tarball_sha256": "\([0-9a-f]\{64\}\)".*/\1/p' "$manifest")
  bytes=$(sed -n 's/^ *"tarball_bytes": \([0-9]*\).*/\1/p' "$manifest")
  size=$(stat -c %s "$tarball")
  if [[ ${#expected} -ne 64 || -z "$bytes" ]]; then
    log "$manifest lacks tarball_sha256/tarball_bytes"; exit 4
  fi
  if [[ "$size" != "$bytes" ]]; then
    actual="size $size B, manifest $bytes B"
  else
    actual=$(sha256sum < "$tarball" | cut -c1-64)
    [[ "$actual" != "$expected" ]] || { log "$tarball matches its manifest (sha256, $((SECONDS - t0)) s)"; return 0; }
    actual="sha256 $actual, manifest $expected"
  fi
  log "$tarball does not match its manifest ($actual): the published artifact is damaged." \
    "It is never replaced in place: move it and $manifest aside, then rebuild"
  exit 4
}

# Wait for another job's publication of the same key (it holds the lock dir).
await_publication() {
  local lock=$1 manifest=$2 waited=0
  log "another job is publishing $(basename "${manifest%.json}") ($lock); waiting for its manifest"
  while [[ -d "$lock" && ! -f "$manifest" ]] && ((waited < ${PACK_VENV_LOCK_WAIT:-1800})); do
    sleep 10
    waited=$((waited + 10))
  done
  if [[ ! -f "$manifest" ]]; then
    log "no manifest after ${waited} s; if no job is publishing this key, remove $lock and rerun"
    exit 3
  fi
}

pack() {
  local venv=$1 name=$2 orig=$3 out=$4 extra=${5:-}
  if [[ "$venv" != "$work_prefix"* || ! -f "$venv/pyvenv.cfg" || ! -L "$venv/bin/python" ]]; then
    log "VENV must be a venv under $work_prefix: $venv"; exit 2
  fi
  if [[ ! "$name" =~ ^[a-z0-9][a-z0-9-]*[a-z0-9]$ || "$out" != "$out_prefix"* || "$orig" != /* ]]; then
    log "bad NAME ($name), OUT_DIR ($out, must be under $out_prefix) or ORIGINAL_PREFIX ($orig)"; exit 2
  fi
  if [[ -e "$venv/.venv-pack" ]]; then
    log "$venv is already packed"; exit 2
  fi
  if [[ -n "$extra" && ! -f "$extra" ]]; then
    log "missing EXTRA_JSON $extra"; exit 2
  fi
  local site version key
  site=$(echo "$venv"/lib/python3.*/site-packages)
  # uv writes version_info; python -m venv writes version.
  version=$(sed -n -e 's/^version_info = //p' -e 's/^version = //p' "$venv/pyvenv.cfg" | head -n 1)
  if [[ ! -d "$site" || -z "$version" ]]; then
    log "cannot find site-packages or the Python version in $venv"; exit 2
  fi
  key=$(key_material "$name" "$version" "$site" | sha256sum | cut -c1-16)
  local tarball="$out/$name-$key.tar.zst" manifest="$out/$name-$key.json" lock="$out/.$name-$key.publishing"
  if [[ -f "$tarball" && -f "$manifest" ]]; then
    verify_published "$tarball" "$manifest"
    log "$tarball is already published (same distributions and recipe); left untouched"
    echo "$tarball"
    return 0
  fi
  mkdir -p "$out"
  if ! mkdir "$lock" 2>/dev/null; then
    await_publication "$lock" "$manifest"
    verify_published "$tarball" "$manifest"
    log "$tarball was published by another job; left untouched"
    echo "$tarball"
    return 0
  fi
  local part="$out/.$name-$key.tar.zst.partial-${SLURM_JOB_ID:-$$}"
  # shellcheck disable=SC2064
  trap "rm -f -- '$part' '$part.json'; rmdir -- '$lock' 2>/dev/null || true" EXIT
  trap 'exit 143' TERM INT
  if [[ -f "$manifest" ]]; then  # published between the check above and the lock
    verify_published "$tarball" "$manifest"
    log "$tarball was published by another job; left untouched"
    echo "$tarball"
    return 0
  fi

  # 1. Build-path references. Bytecode is skipped: it embeds source paths, but
  #    the import system rewrites co_filename when it loads a .pyc.
  mkdir "$venv/.venv-pack"
  printf '%s\n' "$orig" > "$venv/.venv-pack/original-prefix"
  : > "$venv/.venv-pack/prefix-files"
  : > "$venv/.venv-pack/binary-prefix-files"
  local rel
  (cd "$venv" && { grep -rlZF --exclude='*.pyc' --exclude-dir=.venv-pack -- "$orig" . || true; }) |
    while IFS= read -r -d '' rel; do
      if grep -qIF -- "$orig" "$venv/$rel"; then
        printf '%s\0' "$rel" >> "$venv/.venv-pack/prefix-files"
      else
        printf '%s\0' "$rel" >> "$venv/.venv-pack/binary-prefix-files"
      fi
    done

  # 2. Bundle the base interpreter.
  local base_python base_root
  base_python=$(readlink -f "$venv/bin/python")
  base_root=$(dirname "$(dirname "$base_python")")
  if [[ ! -x "$base_python" || ! -f "$base_root/lib/python${version%.*}/os.py" ]]; then
    log "base interpreter $base_python does not look like a standalone CPython"; exit 2
  fi
  local base_refs
  base_refs=$(cd "$venv" && { grep -rlIF --exclude='*.pyc' --exclude-dir=.venv-pack -- "$base_root" . || true; } | grep -vx './pyvenv.cfg' || true)
  if [[ -n "$base_refs" ]]; then
    log "files other than pyvenv.cfg reference $base_root (left as is): $base_refs"
  fi
  copy_tree "$base_root" "$venv/.python"
  ln -sfn "../.python/bin/${base_python##*/}" "$venv/bin/python"
  sed -i "s|^home = .*|home = $orig/.python/bin|" "$venv/pyvenv.cfg"
  printf '%s\0' ./pyvenv.cfg >> "$venv/.venv-pack/prefix-files"

  # 3. Pack: tar | zstd | tee to a partial file on /data, hashing the bytes written.
  local unpacked files sha size
  unpacked=$(du -sb "$venv" | cut -f1)
  files=$(find "$venv" | wc -l)
  local t0=$SECONDS
  sha=$(tar -C "$venv" --sort=name -cf - . | zstd -q -T"$workers" -3 -c | tee "$part" | sha256sum | cut -c1-64)
  size=$(stat -c %s "$part")
  log "packed $files files ($unpacked B) into $size B in $((SECONDS - t0)) s"
  "$venv/.python/bin/python3" - "$part.json" "$name" "$key" "$recipe" "$tarball" "$sha" "$size" \
    "$unpacked" "$files" "$orig" "$venv/.venv-pack" "$base_root" "$version" "$site" "$extra" <<'PY'
import datetime, json, os, pathlib, socket, sys
(out, name, key, recipe, tarball, sha, size, unpacked, files, orig, meta, base_root,
 version, site, extra) = sys.argv[1:]
def listed(path):
    return [p.decode() for p in pathlib.Path(path).read_bytes().split(b"\0") if p]
manifest = {
    "name": name,
    "key": key,
    "recipe": recipe,
    "tarball": tarball,
    "tarball_sha256": sha,
    "tarball_bytes": int(size),
    "compression": "zstd -3",
    "unpacked_bytes": int(unpacked),
    "files": int(files),
    "python": {"version": version, "bundled_from": base_root},
    "distributions": len(list(pathlib.Path(site).glob("*.dist-info"))),
    "original_prefix": orig,
    "prefix_files": listed(os.path.join(meta, "prefix-files")),
    "binary_prefix_files": listed(os.path.join(meta, "binary-prefix-files")),
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "slurm_job": os.environ.get("SLURM_JOB_ID"),
    "node": socket.gethostname(),
    "stage": f"stage_venv {tarball} {key} <VAR>",
}
if extra:
    manifest["source"] = json.loads(pathlib.Path(extra).read_text())
pathlib.Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
PY
  mv -T "$part" "$tarball"
  mv -T "$part.json" "$manifest"
  log "published $tarball (sha256 $sha) and $manifest"
  echo "$tarball"
}

case "${1:-}" in
  copy) [[ $# -eq 3 ]] || { log "usage: pack_venv.sh copy SRC DEST"; exit 2; }; copy_tree "$2" "$3" ;;
  pack) [[ $# -ge 5 && $# -le 6 ]] || { log "usage: pack_venv.sh pack VENV NAME ORIGINAL_PREFIX OUT_DIR [EXTRA_JSON]"; exit 2; }
        shift; pack "$@" ;;
  *) log "usage: pack_venv.sh {copy SRC DEST | pack VENV NAME ORIGINAL_PREFIX OUT_DIR [EXTRA_JSON]}"; exit 2 ;;
esac
