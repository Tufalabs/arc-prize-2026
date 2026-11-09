#!/bin/bash
#SBATCH --job-name=build-env
#SBATCH --account=research
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/data/personal/%u/arc-training-envs/slurm-%j.out
#
# Build a Python environment for cluster jobs from a pushed revision, check it,
# pack it into one immutable artifact on /data, stage it on this node and check
# the staged copy. See docs/environments.md. CPU only; the output directory
# must exist before submission (Slurm does not create it):
#
#   mkdir -p /data/personal/$USER/arc-training-envs
#   sbatch scripts/build_env.sbatch trainer <revision> [options]
#   sbatch scripts/build_env.sbatch harness <revision> [options]
#   sbatch scripts/build_env.sbatch arc2-worker <revision> [options]
#   bash scripts/build_env.sbatch <recipe> <revision> [options] --print-plan
#
# Recipes:
#   trainer  The repository's uv.lock exported with --extra NAME (repeatable,
#            default flash-distillation), installed into a fresh venv, plus each
#            --harness-package NAME (repeatable, default ullm) from the harness
#            source installed with --no-deps. --no-harness-packages installs
#            none. --forbid MODULE (repeatable, default vllm flashinfer platypus
#            duck) must not be importable: the trainer never carries the
#            serving stack. Name: trainer[-<extra>...].
#   harness  The harness repository (the `platypus` submodule) at the commit
#            <revision> pins, or --harness-revision SHA, synced from its own
#            lock: uv sync --locked --no-dev --no-editable --package NAME
#            (--package, default duck). Name: harness, or harness-<package>.
#   arc2-worker
#            The ARC-2 rollout worker paired with the harness environment: the
#            uv project environments/arc2-worker-vllm-fork (duck, platypus and
#            taaf from the `arc2-platypus` submodule, ullm from the `platypus`
#            submodule, both at the commits <revision> pins) synced from its own
#            lock: uv sync --locked --no-dev --no-editable. Its lock must agree
#            with the harness lock on every shared distribution: the worker
#            checks its tokenizer and image processing against the engine's.
#            Name: arc2-worker-vllm-fork.
# Common options:
#   --out DIR            artifact directory on /data (default /data/personal/$USER/arc-training-envs)
#   --repo DIR           fetch-only clone of this repository (default /data/personal/$USER/arc-training-src;
#                        with --print-plan, the checkout holding this script, if it is in one)
#   --harness-clone DIR  existing clone holding the harness revision (default: a fresh
#                        clone of the submodule URL into the job's scratch)
#   --arc2-harness-clone DIR
#                        arc2-worker: the same for the `arc2-platypus` submodule
#   --python PATH        standalone CPython to build on (default: the shared 3.12.12 seed)
#   --no-stage           publish without staging and rechecking on this node
#   --print-plan         print the resolved recipe and exit; builds nothing, needs no Slurm
#
# The artifact is <out>/<name>-<key>.tar.zst plus <name>-<key>.json, the
# manifest with provenance; the key hashes the installed distributions, so an
# unchanged lock gives the same key and an existing artifact is reported, never
# overwritten. Every successful build also writes <out>/receipts/<name>-<key>-<job>.json
# with its own provenance, so a revision whose environment matches an earlier
# build's resolves to it (scripts/resolve_envs.py). scripts/build_envs.sh submits
# all three recipes. Jobs stage an artifact with scripts/stage_venv.sh. The helpers
# (pack_venv.sh, stage_venv.sh, env_checks.py) come from <revision> itself.
set -euo pipefail

log() { echo "build_env: $*" >&2; }
die() { log "$*"; exit 2; }
usage() {
  if [[ -f "$0" ]]; then sed -n '/^# Build a Python/,/^# (pack_venv.sh, stage_venv.sh/s/^# \{0,1\}//p' "$0"
  else echo "usage: see the header of scripts/build_env.sbatch and docs/environments.md"; fi
}
join() { local IFS=$1; shift; printf '%s' "$*"; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum | cut -c1-64; else shasum -a 256 | cut -c1-64; fi; }
# The vLLM commit a uv.lock (stdin) pins as a git source, if any.
fork_commit() { sed -n -e '/^name = "vllm"$/,/^$/s/^source = { git = "[^"]*#\([0-9a-f]\{40\}\)".*/\1/p'; }
# Submodule NAME's gitlink commit and .gitmodules URL at $revision of $repo (empty if absent).
submodule_pin() { git -C "$repo" ls-tree "$revision" "$1" | awk '$1 == "160000" {print $3}'; }
submodule_url() { git -C "$repo" config --blob "$revision:.gitmodules" --get "submodule.$1.url" || true; }

recipe=${1:-}
case "$recipe" in
  trainer|harness|arc2-worker) shift ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; die "first argument must be a recipe: trainer, harness or arc2-worker" ;;
esac
[[ $# -gt 0 && "$1" != -* ]] || die "missing <revision>: a commit of this repository that has been pushed"
revision_arg=$1
shift

user=${USER:-$(id -un)}
extras=(flash-distillation) extras_given=0
harness_packages=(ullm) harness_packages_given=0
forbid=(vllm flashinfer platypus duck) forbid_given=0
package=duck harness_revision="" harness_clone="" arc2_clone="" stage=1 print_plan=0 repo_given=0
out="/data/personal/$user/arc-training-envs"
repo="/data/personal/$user/arc-training-src"
seed=/data/shared/research/arc3_uv_seed
python="$seed/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
submodule=platypus  # the harness submodule; its packages/<name> are the workspace members
arc2_submodule=arc2-platypus  # the ARC-2 harness submodule (arc2-worker)
project=arc2-worker-vllm-fork  # arc2-worker: the uv project under environments/

only() { [[ "$recipe" == "$1" ]] || die "$2 applies to the $1 recipe"; }
while [[ $# -gt 0 ]]; do
  opt=$1
  case "$opt" in
    --no-stage) stage=0; shift; continue ;;
    --print-plan) print_plan=1; shift; continue ;;
    -h|--help) usage; exit 0 ;;
    --no-harness-packages) only trainer "$opt"; harness_packages=(); harness_packages_given=1; shift; continue ;;
    --extra|--harness-package|--forbid|--harness-revision|--package|--harness-clone|--arc2-harness-clone) ;;
    --out|--repo|--python) ;;
    *) die "unknown option $opt (see --help)" ;;
  esac
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$opt needs a value"
  value=$2
  shift 2
  case "$opt" in
    --extra) only trainer "$opt"; ((extras_given)) || extras=(); extras_given=1; extras+=("$value") ;;
    --harness-package)
      only trainer "$opt"; ((harness_packages_given)) || harness_packages=(); harness_packages_given=1
      harness_packages+=("$value") ;;
    --forbid) only trainer "$opt"; ((forbid_given)) || forbid=(); forbid_given=1; forbid+=("$value") ;;
    --harness-revision) only harness "$opt"; harness_revision=$value ;;
    --package) only harness "$opt"; package=$value ;;
    --harness-clone) harness_clone=$value ;;
    --arc2-harness-clone) only arc2-worker "$opt"; arc2_clone=$value ;;
    --out) out=${value%/} ;;
    --repo) repo=${value%/}; repo_given=1 ;;
    --python) python=$value ;;
  esac
done
# A plan run from a checkout reads that checkout, as scripts/build_envs.sh does.
# A submitted build runs a copy of this script and keeps the fetch-only clone.
if ((print_plan && !repo_given)); then
  here=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || here=""
  if [[ -n "$here" ]] && top=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) &&
    [[ "$top/scripts" -ef "$here" ]]; then
    repo=$top
    log "planning the checkout holding this script, $repo (--repo DIR plans another clone)"
  fi
fi

if [[ "$recipe" != trainer ]]; then harness_packages=() forbid=() extras=(); fi
name_re='^[a-z0-9]([a-z0-9-]*[a-z0-9])?$'
module_re='^[A-Za-z_][A-Za-z0-9_]*$'
for value in ${extras[@]+"${extras[@]}"} ${harness_packages[@]+"${harness_packages[@]}"} "$package"; do
  [[ "$value" =~ $name_re ]] || die "bad name '$value': use the normalized form (lowercase letters, digits, -)"
done
for value in ${forbid[@]+"${forbid[@]}"}; do
  [[ "$value" =~ $module_re ]] || die "bad module name '$value' for --forbid"
  for pkg in ${harness_packages[@]+"${harness_packages[@]}"}; do
    [[ "${pkg//-/_}" != "$value" ]] || die "--harness-package $pkg is also forbidden (--forbid $value)"
  done
done
if [[ -n "$(printf '%s\n' ${extras[@]+"${extras[@]}"} | sort | uniq -d)" ]]; then die "an --extra is repeated"; fi
if [[ -n "$(printf '%s\n' ${harness_packages[@]+"${harness_packages[@]}"} | sort | uniq -d)" ]]; then
  die "a --harness-package is repeated"
fi
[[ -z "$harness_revision" || "$harness_revision" =~ ^[0-9a-f]{40}$ ]] || die "--harness-revision must be a full 40-hex commit"
[[ "$out" == /data/* ]] || die "--out must be a directory under /data (got $out)"
for value in "$repo" "$python" ${harness_clone:+"$harness_clone"} ${arc2_clone:+"$arc2_clone"}; do
  [[ "$value" == /* ]] || die "paths must be absolute: $value"
done

if ((!print_plan)); then
  # The build runs on a compute node inside the allocation, never on the login node.
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "submit with sbatch (see --help); --print-plan shows the recipe without building"
  # --export=ALL carries the submitting shell's environment: drop interpreter,
  # uv and cache settings so the build depends only on the lock and this
  # script, and so job-env.sh applies this job's scratch and per-user caches.
  for var in $(compgen -e); do
    case "$var" in UV_*|PYTHON*|VIRTUAL_ENV|CONDA_PREFIX|PIP_*|STAGE_VENV_ROOT) unset "$var" ;; esac
  done
  unset HF_HOME TORCH_HOME XDG_CACHE_HOME SCRATCH_DIR SLURM_TMPDIR TMPDIR TEMPDIR
  . /etc/kander/job-env.sh
fi

# ---- Resolve the recipe (also all of --print-plan) ---------------------------
[[ -d "$repo" ]] || die "fetch-only clone $repo not found. Create it once (on the login node):
  git clone --no-checkout <url of this repository> $repo
then before each build: git -C $repo fetch origin   (or pass --repo DIR)"
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || die "$repo is not a git repository"
revision=$(git -C "$repo" rev-parse --verify --quiet "$revision_arg^{commit}") ||
  die "$revision_arg is not a commit in $repo: push it, then git -C $repo fetch origin"
[[ "$revision_arg" == "$revision" ]] || log "note: $revision_arg resolves to $revision in $repo"
repo_url=$(git -C "$repo" config --get remote.origin.url || echo unknown)
repo_url=$(printf '%s' "$repo_url" | sed -E 's#://[^/@]+@#://#')  # never record credentials
lock_sha=$(git -C "$repo" cat-file blob "$revision:uv.lock" | sha256) || die "no uv.lock at $revision"
pin=$(submodule_pin "$submodule")
harness_url=$(submodule_url "$submodule")

need_harness=0
if [[ "$recipe" != trainer || ${#harness_packages[@]} -gt 0 ]]; then need_harness=1; fi
harness_rev="" harness_rev_source="" harness_lock_sha="" fork_revision=""
if ((need_harness)); then
  [[ -n "$pin" && -n "$harness_url" ]] || die "$revision has no $submodule submodule (gitlink and .gitmodules url; see docs/environments.md#building)"
  if [[ -n "$harness_revision" ]]; then
    harness_rev=$harness_revision harness_rev_source=--harness-revision
  else
    harness_rev=$pin harness_rev_source="submodule pin"
  fi
  if [[ -n "$harness_clone" ]]; then
    git -C "$harness_clone" cat-file -e "$harness_rev^{commit}" 2>/dev/null ||
      die "$harness_rev is not in $harness_clone: git -C $harness_clone fetch origin"
    for pkg in ${harness_packages[@]+"${harness_packages[@]}"}; do
      git -C "$harness_clone" cat-file -e "$harness_rev:packages/$pkg/pyproject.toml" 2>/dev/null ||
        die "--harness-package $pkg: no packages/$pkg at $harness_rev"
    done
    harness_lock_sha=$(git -C "$harness_clone" cat-file blob "$harness_rev:uv.lock" | sha256) ||
      die "no uv.lock at $harness_rev"
    fork_revision=$(git -C "$harness_clone" cat-file blob "$harness_rev:uv.lock" | fork_commit)
  fi
fi
# arc2-worker: the project's own lock, and the ARC-2 harness at its pin.
project_lock_sha="" arc2_pin="" arc2_url=""
if [[ "$recipe" == arc2-worker ]]; then
  project_dir="environments/$project"
  git -C "$repo" cat-file -e "$revision:$project_dir/pyproject.toml" 2>/dev/null &&
    git -C "$repo" cat-file -e "$revision:$project_dir/uv.lock" 2>/dev/null ||
    die "$revision has no $project_dir/pyproject.toml and uv.lock"
  project_lock_sha=$(git -C "$repo" cat-file blob "$revision:$project_dir/uv.lock" | sha256)
  # The engine the worker renders with; the job checks it against the harness lock's.
  fork_revision=$(git -C "$repo" cat-file blob "$revision:$project_dir/uv.lock" | fork_commit)
  arc2_pin=$(submodule_pin "$arc2_submodule")
  arc2_url=$(submodule_url "$arc2_submodule")
  [[ -n "$arc2_pin" && -n "$arc2_url" ]] ||
    die "$revision has no $arc2_submodule submodule (gitlink and .gitmodules url; see docs/environments.md#building)"
  if [[ -n "$harness_clone" ]]; then
    git -C "$harness_clone" cat-file -e "$harness_rev:packages/ullm/pyproject.toml" 2>/dev/null ||
      die "no packages/ullm at harness $harness_rev"
  fi
  if [[ -n "$arc2_clone" ]]; then
    git -C "$arc2_clone" cat-file -e "$arc2_pin^{commit}" 2>/dev/null ||
      die "$arc2_pin is not in $arc2_clone: git -C $arc2_clone fetch origin"
    for pkg in duck platypus taaf; do
      git -C "$arc2_clone" cat-file -e "$arc2_pin:packages/$pkg/pyproject.toml" 2>/dev/null ||
        die "no packages/$pkg at $arc2_submodule $arc2_pin"
    done
  fi
fi

if [[ "$recipe" == trainer ]]; then
  sorted_extras=()
  while IFS= read -r value; do [[ -z "$value" ]] || sorted_extras+=("$value"); done \
    < <(printf '%s\n' ${extras[@]+"${extras[@]}"} | sort)
  name=$(join - trainer ${sorted_extras[@]+"${sorted_extras[@]}"})
  extra_args=()
  for value in ${sorted_extras[@]+"${sorted_extras[@]}"}; do extra_args+=(--extra "$value"); done
  build_steps="uv export --frozen --no-dev $(join ' ' ${extra_args[@]+"${extra_args[@]}"}) --no-emit-project"
  build_steps+=" --no-emit-workspace; uv venv --relocatable; uv pip install --compile-bytecode -r <export>"
  for pkg in ${harness_packages[@]+"${harness_packages[@]}"}; do
    build_steps+="; uv pip install --no-deps --compile-bytecode <harness>/packages/$pkg"
  done
  check_steps="lock export = installed; absent: $(join ' ' ${forbid[@]+"${forbid[@]}"})"
  if ((${#harness_packages[@]})); then check_steps+="; imports: $(join ' ' "${harness_packages[@]//-/_}")"; fi
  for pkg in ${harness_packages[@]+"${harness_packages[@]}"}; do
    [[ "$pkg" != ullm ]] || check_steps+=" ullm.media"
  done
  check_steps+="; tests/distillation/import_probes/*"
elif [[ "$recipe" == arc2-worker ]]; then
  name=$project
  build_steps="uv sync --locked --no-dev --no-editable --compile-bytecode in $project_dir"
  build_steps+=" (with $arc2_submodule and $submodule at their pins)"
  check_steps="the project lock agrees with the harness lock on every shared distribution; lock export ="
  check_steps+=" installed (duck, platypus, taaf, ullm imported); vllm reports the locked fork commit;"
  check_steps+=" harness_support.arc2.worker imports, scripts/platypus_rollout_worker.py --help; the trainer's"
  check_steps+=" engine presets exist; both ARC-2 datasets load at their sizes and equal the submodule's files"
else
  name=harness
  [[ "$package" == duck ]] || name="harness-$package"
  build_steps="uv sync --locked --no-dev --no-editable --compile-bytecode --package $package"
  check_steps="lock export = installed (git commits, workspace members imported); vllm reports the locked"
  check_steps+=" fork commit; training.server_endpoints, harness_support.arc3.worker,"
  check_steps+=" scripts/platypus_rollout_worker.py import; readiness CLI; native_vllm_server --help only with a GPU"
fi
existing=()
if [[ -d "$out" ]]; then
  for path in "$out/$name"-*.tar.zst; do
    base=${path##*/}
    if [[ "$base" =~ ^$name-[0-9a-f]{16}\.tar\.zst$ && -f "${path%.tar.zst}.json" ]]; then existing+=("$base"); fi
  done
fi
uv=$(command -v uv || echo "$seed/uv")

print_plan_lines() {
  printf '%s=%s\n' recipe "$recipe" revision "$revision" repo "$repo" repository "$repo_url" \
    uv_lock_sha256 "$lock_sha"
  if [[ "$recipe" == trainer ]]; then
    printf '%s=%s\n' extras "$(join ' ' ${sorted_extras[@]+"${sorted_extras[@]}"})" \
      harness_packages "$(join ' ' ${harness_packages[@]+"${harness_packages[@]}"})" \
      forbid "$(join ' ' ${forbid[@]+"${forbid[@]}"})"
  elif [[ "$recipe" == arc2-worker ]]; then
    printf '%s=%s\n' project "$project_dir" project_uv_lock_sha256 "$project_lock_sha"
  else
    printf '%s=%s\n' package "$package"
  fi
  if ((need_harness)); then
    printf '%s=%s\n' harness_submodule_pin "$pin" harness_revision "$harness_rev" \
      harness_revision_source "$harness_rev_source" harness_repository "$harness_url" \
      harness_source "${harness_clone:-fresh clone of $harness_url into job scratch}" \
      harness_uv_lock_sha256 "${harness_lock_sha:-read in the job}"
  fi
  if [[ "$recipe" == arc2-worker ]]; then
    printf '%s=%s\n' arc2_harness_submodule_pin "$arc2_pin" arc2_harness_repository "$arc2_url" \
      arc2_harness_source "${arc2_clone:-fresh clone of $arc2_url into job scratch}" \
      vllm_fork_revision "${fork_revision:-none}"
  elif [[ "$recipe" == harness ]]; then
    printf '%s=%s\n' vllm_fork_revision "${fork_revision:-read in the job}"
  fi
  printf '%s=%s\n' name "$name" artifact "$out/$name-<key>.tar.zst" out "$out" python "$python" uv "$uv" \
    stage "$( ((stage)) && echo yes || echo no)" existing "$(join ' ' ${existing[@]+"${existing[@]}"})" \
    build "$build_steps" checks "$check_steps"
}
if ((print_plan)); then
  print_plan_lines
  exit 0
fi

# ---- Build -------------------------------------------------------------------
uid=$(id -u)
cpus=${SLURM_CPUS_PER_TASK:-1}
work="${SCRATCH_DIR:?job-env.sh did not provide SCRATCH_DIR}/build-env"
venv="$work/env/$name"  # its own directory: names may match source directories
# A stage root of this job's own, for an artifact it publishes (below); removed at exit.
private_stage=""
CHILD=""
remove_private_stage() {
  if [[ -n "$private_stage" && "$private_stage" == "/cache/users/$uid/build-env-stage-$SLURM_JOB_ID" ]]; then
    rm -rf -- "$private_stage" && log "removed $private_stage"
  fi
  private_stage=""
}
cleanup() {
  local status=$?
  if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM "$CHILD" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
    kill -KILL "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
  cd /
  remove_private_stage
  [[ "$work" == /scratch/* ]] && rm -rf -- "$work"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
# Run a command as a waited child so TERM reaches cleanup promptly.
run() {
  local status=0
  "$@" &
  CHILD=$!
  wait "$CHILD" || status=$?
  CHILD=""
  return "$status"
}

[[ "${UV_CACHE_DIR:-}" == "/cache/users/$uid/uv" ]] ||
  die "UV_CACHE_DIR=${UV_CACHE_DIR:-unset}; job-env.sh should provide /cache/users/$uid/uv"
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy  # /cache and /scratch are separate filesystems
export UV_CONCURRENT_INSTALLS=$cpus UV_CONCURRENT_BUILDS=$cpus CMAKE_BUILD_PARALLEL_LEVEL=$cpus MAX_JOBS=$cpus
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=30}"
log "job $SLURM_JOB_ID on $(hostname), $cpus CPUs; work=$work UV_CACHE_DIR=$UV_CACHE_DIR TMPDIR=${TMPDIR:-unset}"
print_plan_lines >&2
[[ -x "$uv" ]] || die "no uv on PATH and none at $uv"
uv_version=$("$uv" --version)
log "$uv_version"

mkdir -p "$work/src" "$work/env"
run timeout 10m bash -c 'set -euo pipefail; git -C "$1" archive "$2" | tar -x -C "$3"' _ "$repo" "$revision" "$work/src"
tools="$work/src/scripts"
for f in pack_venv.sh stage_venv.sh env_checks.py; do
  [[ -f "$tools/$f" ]] || die "$revision has no scripts/$f: build from a revision that carries the environment tooling"
done
builder_sha=$(sha256 < "$0")
builder_matches=false
if cmp -s "$0" "$tools/build_env.sbatch"; then builder_matches=true; else
  log "warning: the submitted build_env.sbatch differs from $revision's; the manifest records builder_matches_revision=false"
fi

# The URL to read a GitHub repository with from this node: URL itself, or the
# same repository over the other transport (HTTPS credentials vs SSH key).
readable_url() {
  local url=$1 path="" candidate
  case "$url" in
    https://github.com/*) path=${url#https://github.com/}; set -- "$url" "ssh://git@github.com/$path" ;;
    ssh://git@github.com/*) path=${url#ssh://git@github.com/}; set -- "$url" "https://github.com/$path" ;;
    git@github.com:*) path=${url#git@github.com:}; set -- "$url" "https://github.com/$path" ;;
    *) set -- "$url" ;;
  esac
  for candidate in "$@"; do
    if timeout 90 git ls-remote --exit-code "$candidate" HEAD >/dev/null 2>&1; then echo "$candidate"; return 0; fi
  done
  return 1
}
# uv fetches git requirements with git: route each GitHub repository the
# install names to a transport this user can read (GIT_CONFIG_* insteadOf for
# this job only; the lock is unchanged).
route_git_sources() {
  local url readable n=${GIT_CONFIG_COUNT:-0}
  while IFS= read -r url; do
    readable=$(readable_url "$url") || die "cannot read $url from $(hostname) over HTTPS or SSH; check your GitHub access"
    if [[ "$readable" != "$url" ]]; then
      export "GIT_CONFIG_KEY_$n=url.$readable.insteadOf" "GIT_CONFIG_VALUE_$n=$url"
      n=$((n + 1))
      log "git source $url is read as $readable"
    fi
  done < <(grep -oE '(git = "|git\+)(https://|ssh://git@)github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' "$1" |
    sed -E 's/^(git = "|git\+)//' | sort -u)
  export GIT_CONFIG_COUNT=$n
}

# Extract repository URL at REV into DEST, from CLONE if given, else from a
# fresh bare clone named NAME in scratch (CLONE_OPTION names the option).
fetch_tree() {
  local url=$1 rev=$2 clone=$3 dest=$4 name=$5 clone_option=$6 fetch_url
  mkdir -p "$dest"
  if [[ -z "$clone" ]]; then
    clone="$work/$name.git"
    fetch_url=$(readable_url "$url") || die "cannot read $url over HTTPS or SSH; pass $clone_option"
    run timeout 5m git init -q --bare "$clone"
    if ! run timeout 20m git -C "$clone" fetch -q --depth 1 --no-tags "$fetch_url" "$rev"; then
      log "fetching $rev by id failed; fetching all branches of $fetch_url"
      run timeout 40m git -C "$clone" fetch -q --no-tags "$fetch_url" '+refs/heads/*:refs/remotes/origin/*'
    fi
  fi
  run timeout 10m bash -c 'set -euo pipefail; git -C "$1" cat-file -e "$2^{commit}"
    git -C "$1" archive "$2" | tar -x -C "$3"' _ "$clone" "$rev" "$dest"
}

# arc2-worker builds its project inside a tree laid out as the repository is
# (environments/<project> beside both submodules), so its relative sources resolve.
harness_src="$work/harness-src"
[[ "$recipe" != arc2-worker ]] || harness_src="$work/project/$submodule"
if ((need_harness)); then
  fetch_tree "$harness_url" "$harness_rev" "$harness_clone" "$harness_src" harness --harness-clone
  harness_lock_sha=$(sha256 < "$harness_src/uv.lock")
  harness_fork=$(fork_commit < "$harness_src/uv.lock")
  [[ "$recipe" == arc2-worker ]] || fork_revision=$harness_fork
  log "harness $harness_rev ($harness_rev_source), uv.lock sha256 $harness_lock_sha${harness_fork:+, vLLM fork $harness_fork}"
fi
if [[ "$recipe" == arc2-worker ]]; then
  fetch_tree "$arc2_url" "$arc2_pin" "$arc2_clone" "$work/project/$arc2_submodule" arc2-harness --arc2-harness-clone
  mkdir -p "$work/project/environments"
  cp -R "$work/src/$project_dir" "$work/project/$project_dir"
  log "$arc2_submodule $arc2_pin (submodule pin); $project_dir uv.lock sha256 $project_lock_sha"
fi

# Build on a scratch copy of the base interpreter (no per-import reads from /data).
python_real=$(readlink -f "$python")
python_root=$(dirname "$(dirname "$python_real")")
[[ -x "$python_real" ]] || die "no interpreter at $python"
run timeout 15m bash "$tools/pack_venv.sh" copy "$python_root" "$work/python"
build_python="$work/python/bin/${python_real##*/}"

if [[ "$recipe" == trainer ]]; then
  cd "$work/src"
  run timeout 10m "$uv" export --frozen --no-dev ${extra_args[@]+"${extra_args[@]}"} --no-emit-project \
    --no-emit-workspace --format requirements-txt --no-header --output-file "$work/requirements.txt" >/dev/null
  cd "$work"
  forbidden_re="^($(join '|' ${forbid[@]+"${forbid[@]//_/[-_.]}"}))([-_.][A-Za-z0-9_.-]*)? *(==| @ )"
  if ((${#forbid[@]})) && grep -iE "$forbidden_re" "$work/requirements.txt"; then
    die "the lock export for $(join ' ' "${sorted_extras[@]}") contains forbidden packages (above)"
  fi
  route_git_sources "$work/requirements.txt"
  run timeout 10m "$uv" venv --relocatable --prompt "$name" --python "$build_python" "$venv"
  run timeout 90m "$uv" pip install --python "$venv/bin/python" --compile-bytecode -r "$work/requirements.txt"
  for pkg in ${harness_packages[@]+"${harness_packages[@]}"}; do
    [[ -f "$harness_src/packages/$pkg/pyproject.toml" ]] || die "no packages/$pkg at harness $harness_rev"
    run timeout 20m "$uv" pip install --python "$venv/bin/python" --no-deps --compile-bytecode \
      "$harness_src/packages/$pkg"
  done
elif [[ "$recipe" == arc2-worker ]]; then
  # The worker verifies its tokenizer and image processing against the
  # harness engine's: refuse a lock that disagrees before installing anything.
  run timeout 5m "$build_python" -I -B "$tools/env_checks.py" lock-agrees \
    "$work/project/$project_dir/uv.lock" "$harness_src/uv.lock"
  route_git_sources "$work/project/$project_dir/uv.lock"
  cd "$work/project/$project_dir"
  run timeout 120m env UV_PROJECT_ENVIRONMENT="$venv" "$uv" sync --locked --no-dev --no-editable \
    --compile-bytecode --python "$build_python"
  run timeout 10m "$uv" export --frozen --no-dev --no-editable --no-emit-project \
    --format requirements-txt --no-header --output-file "$work/requirements.txt" >/dev/null
  cd "$work"
else
  route_git_sources "$harness_src/uv.lock"
  cd "$harness_src"
  run timeout 120m env UV_PROJECT_ENVIRONMENT="$venv" "$uv" sync --locked --no-dev --no-editable \
    --compile-bytecode --package "$package" --python "$build_python"
  run timeout 10m "$uv" export --frozen --no-dev --no-editable --package "$package" \
    --format requirements-txt --no-header --output-file "$work/requirements.txt" >/dev/null
  cd "$work"
fi
log "built $venv: $(du -sh "$venv" | cut -f1), $(find "$venv" | wc -l) files"

gpu_visible() { command -v nvidia-smi >/dev/null && timeout 60 nvidia-smi -L 2>/dev/null | grep -q '^GPU '; }
# If the exported requirements pin vLLM as a git source, check that the
# installed vLLM reports that commit; "$@" is the isolated env_checks.py command.
check_locked_vllm() {
  local commit
  commit=$(sed -n -E 's/^vllm @ git\+[^@]+@([0-9a-f]{40}).*/\1/p' "$work/requirements.txt")
  if [[ -n "$commit" ]]; then run timeout 10m "$@" reports-commit vllm "$commit"; fi
}
# The recipe's checks against VENV (the build tree, then the staged copy). The
# environment checks run isolated from the source tree (-I, neutral working
# directory); the repository checks put the source on PYTHONPATH.
check_env() {
  local venv=$1 py=$1/bin/python dir rc=0 probe
  dir=$(mktemp -d "$work/checks.XXXXXX")
  local cpu=(PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 FLA_TILELANG=0
    FLA_CONV_BACKEND=triton TRITON_CACHE_DIR="$dir/triton" TORCHINDUCTOR_CACHE_DIR="$dir/inductor")
  local isolated=(env -u PYTHONPATH "${cpu[@]}" "$py" -I -B "$tools/env_checks.py")
  local repo_env=(env "${cpu[@]}" PYTHONPATH="$work/src")
  log "== checks: $venv"
  cd "$dir"
  if [[ "$recipe" == trainer ]]; then
    local allow=() modules=()
    for pkg in ${harness_packages[@]+"${harness_packages[@]}"}; do
      allow+=(--allow-extra "$pkg")
      modules+=("${pkg//-/_}")
      [[ "$pkg" != ullm ]] || modules+=(ullm.media)  # the trainer's multimodal path imports it
    done
    run timeout 10m "${isolated[@]}" lock "$work/requirements.txt" ${allow[@]+"${allow[@]}"}
    if ((${#forbid[@]})); then run timeout 10m "${isolated[@]}" absent "${forbid[@]}"; fi
    if ((${#modules[@]})); then run timeout 10m "${isolated[@]}" imports "${modules[@]}"; fi
    cd "$work/src"
    for probe in tests/distillation/import_probes/*.py; do
      probe=${probe##*/}
      [[ "$probe" != __init__.py ]] || continue
      run timeout 15m "${repo_env[@]}" "$py" -m "tests.distillation.import_probes.${probe%.py}"
      log "import probe ${probe%.py} ok"
    done
  elif [[ "$recipe" == arc2-worker ]]; then
    run timeout 10m "${isolated[@]}" lock "$work/requirements.txt" --base "$work/project/$project_dir" --import-local
    run timeout 10m "${isolated[@]}" imports duck.agent duck.presets taaf.datasets ullm.media \
      ullm.backends.engine_proxy
    check_locked_vllm "${isolated[@]}"
    cd "$work/src"
    # The rollout worker as the trainer spawns it (scripts/ is its sys.path[0]).
    run timeout 10m "${repo_env[@]}" "$py" scripts/platypus_rollout_worker.py --help >/dev/null
    run timeout 10m "${repo_env[@]}" PYTHONPATH="$work/src:$work/src/scripts" "$py" -c '
import sys
from pathlib import Path
from types import SimpleNamespace
import serving_contract, training_backend  # noqa: F401
import taaf
from duck import presets
from taaf.datasets import load
from harness_support.arc2.dataset import DATASET_SIZES
from training.harness import get_harness, harness_engine
get_harness("arc2")
engines = {harness_engine(SimpleNamespace(weight_mode=mode)) for mode in ("lora", "full_weight", "flash")}
missing = engines - set(presets.ENGINES)
assert not missing, f"duck has no engine preset {sorted(missing)}"
pinned = Path(sys.argv[1])
for name, size in DATASET_SIZES.items():
    installed = Path(taaf.__file__).parent / "data" / f"{name}.npz"
    assert installed.read_bytes() == (pinned / f"{name}.npz").read_bytes(), f"{installed} differs from the pin"
    assert len(load(name)) == size, f"{name}: expected {size} tasks"
print(f"ARC-2 worker modules import; engines {sorted(engines)}; datasets {DATASET_SIZES}:", sys.prefix)' \
      "$work/project/$arc2_submodule/packages/taaf/src/taaf/data"
  else
    run timeout 10m "${isolated[@]}" lock "$work/requirements.txt" --base "$harness_src" --import-local
    check_locked_vllm "${isolated[@]}"
    cd "$work/src"
    run timeout 10m "${repo_env[@]}" "$py" -c '
import importlib.util, sys
import harness_support.arc3.worker, training.server_endpoints  # noqa: F401
spec = importlib.util.spec_from_file_location("platypus_rollout_worker", "scripts/platypus_rollout_worker.py")
spec.loader.exec_module(importlib.util.module_from_spec(spec))
print("repository modules import:", sys.prefix)'
    # The readiness CLI against a closed port: exit 1 and no output.
    run timeout 60 "${repo_env[@]}" "$py" -m training.server_endpoints http://127.0.0.1:1/health \
      > "$dir/readiness.out" 2>&1 || rc=$?
    [[ $rc == 1 && ! -s "$dir/readiness.out" ]] || { cat "$dir/readiness.out" >&2; die "readiness CLI returned $rc"; }
    log "readiness CLI ok (unreachable server -> 1)"
    if gpu_visible; then
      run timeout 10m env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work/src" \
        "$py" -m training.native_vllm_server --help >/dev/null  # the GPU stays visible
      log "native_vllm_server --help ok"
    else
      log "skipped: python -m training.native_vllm_server --help: vLLM must detect a GPU to build its" \
        "argument parser, and this CPU allocation has none"
    fi
  fi
  cd "$work"
  [[ "$dir" == "$work"/checks.* ]] && rm -rf -- "$dir"
  log "== checks passed: $venv"
}
check_env "$venv"

# ---- Provenance, pack, publish -----------------------------------------------
json_list() { local item sep="" text="["; for item; do text+="$sep\"$item\""; sep=","; done; printf '%s]' "$text"; }
fields=(builder=scripts/build_env.sbatch "builder_sha256=$builder_sha" "builder_matches_revision:=$builder_matches"
  "recipe=$recipe" "repository=$repo_url" "revision=$revision" "uv_lock_sha256=$lock_sha"
  "requirements_sha256=$(sha256 < "$work/requirements.txt")" "uv=$uv_version" "python_source=$python"
  "slurm_job=$SLURM_JOB_ID" "node=$(hostname)")
if [[ "$recipe" == trainer ]]; then
  fields+=("extras:=$(json_list ${sorted_extras[@]+"${sorted_extras[@]}"})"
    "harness_packages:=$(json_list ${harness_packages[@]+"${harness_packages[@]}"})"
    "forbidden_modules:=$(json_list ${forbid[@]+"${forbid[@]}"})" "install=$build_steps")
elif [[ "$recipe" == arc2-worker ]]; then
  fields+=("project=$project_dir" "project_uv_lock_sha256=$project_lock_sha" "install=$build_steps"
    "vllm_fork_revision=$fork_revision" "arc2_harness_repository=$arc2_url" "arc2_harness_revision=$arc2_pin"
    "arc2_harness_submodule_pin=$arc2_pin")
else
  fields+=("package=$package" "install=$build_steps" "vllm_fork_revision=$fork_revision")
fi
if ((need_harness)); then
  fields+=("harness_repository=$harness_url" "harness_revision=$harness_rev" "harness_revision_source=$harness_rev_source"
    "harness_submodule_pin=$pin" "harness_uv_lock_sha256=$harness_lock_sha")
fi
run timeout 5m env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$venv/bin/python" -I -B "$tools/env_checks.py" \
  provenance "$work/provenance.json" "${fields[@]}"
run timeout 60m bash "$tools/pack_venv.sh" pack "$venv" "$name" "$venv" "$out" "$work/provenance.json" \
  > "$work/tarball.txt"
tarball=$(cat "$work/tarball.txt")
key=${tarball%.tar.zst}
key=${key##*-}
manifest=${tarball%.tar.zst}.json
read -r built_by built_from size < <("$build_python" -c '
import json, sys
m = json.load(open(sys.argv[1]))
print(m.get("slurm_job"), m.get("source", {}).get("revision"), m["tarball_bytes"])' "$manifest")
if [[ "$built_by" == "$SLURM_JOB_ID" ]]; then
  log "published $tarball ($size bytes)"
else
  # pack_venv.sh checked the existing tarball against its manifest's sha256.
  log "unchanged: $tarball already exists (job $built_by, revision $built_from); nothing written"
fi

if ((stage)); then
  stage_root="/cache/users/$uid/venvs"  # stage_venv's default, where jobs stage from
  if [[ "$built_by" == "$SLURM_JOB_ID" ]]; then
    # Unpack what was just written: the shared root may already hold this key
    # (staged from another root or an earlier publication), and a warm stage
    # would check that copy instead.
    private_stage="/cache/users/$uid/build-env-stage-$SLURM_JOB_ID"
    stage_root=$private_stage
    log "staging the new artifact into $stage_root (this job's own; removed at exit)"
  fi
  run timeout -k 30s 30m env STAGE_VENV_ROOT="$stage_root" bash -c \
    'set -euo pipefail; . "$1"; stage_venv_print "$2" STAGED_PYTHON' \
    _ "$tools/stage_venv.sh" "$tarball" > "$work/staged-python"
  staged=$(cat "$work/staged-python")
  check_env "${staged%/bin/python}"
  remove_private_stage
else
  log "--no-stage: not staged on this node"
fi
# A receipt records that this build's identity (this revision's locks and pins)
# installs the artifact, published now or by an earlier build of the same key:
# scripts/resolve_envs.py matches revisions to artifacts by manifests and receipts.
receipt="$out/receipts/$name-$key-$SLURM_JOB_ID.json"
if mkdir -p "$out/receipts" && run timeout 2m "$build_python" -I -B -c '
import datetime, json, os, sys
provenance, name, key, target = sys.argv[1:]
record = {"name": name, "key": key, "tarball": f"{name}-{key}.tar.zst",
          "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "slurm_job": os.environ.get("SLURM_JOB_ID"), "source": json.load(open(provenance))}
with open(target + ".tmp", "w") as handle:
    handle.write(json.dumps(record, indent=2) + "\n")
os.replace(target + ".tmp", target)' "$work/provenance.json" "$name" "$key" "$receipt"; then
  log "receipt: $receipt"
else
  log "warning: no receipt written to $out/receipts; scripts/resolve_envs.py may not match $revision to this artifact"
fi
log "artifact: $tarball"
log "manifest: $manifest"
log "key: $key"
log "in a job: . scripts/stage_venv.sh; stage_venv $tarball $key <VAR>   (exports VAR=<staged venv>/bin/python)"
echo "BUILD_ENV_RESULT recipe=$recipe name=$name key=$key tarball=$tarball bytes=$size staged=$( ((stage)) && echo yes || echo no)"
