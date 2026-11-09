# Job environments on Kander

RL jobs run from prebuilt Python environments, never from a venv built inside
the job. Each environment is one immutable artifact on `/data`, a
`<name>-<key>.tar.zst` with a JSON manifest beside it, built by
`scripts/build_env.sbatch` and unpacked onto each node's `/cache` by
`scripts/stage_venv.sh`.

## The environments

| Recipe | Built from | Runs | Name |
|---|---|---|---|
| `trainer` | this repository's `uv.lock`, exported with the selected extras (default `flash-distillation`), plus harness packages installed without dependencies (default `ullm`) | the learner (`ARC3_FLASH_PYTHON`) | `trainer-<extras>` |
| `harness` | the harness repository (the `platypus` submodule, arc3-platypus) at the commit this repository pins, synced from the harness's own lock with `uv sync --locked --no-dev --no-editable --package duck` | serving engines and ARC-3 rollout workers (`PLATYPUS_VENV`) | `harness` |
| `arc2-worker` | this repository's `environments/arc2-worker-vllm-fork` project: duck, platypus and taaf from the `arc2-platypus` submodule and ullm from the `platypus` submodule, at the commits this repository pins, synced from the project's own lock with `uv sync --locked --no-dev --no-editable` | ARC-2 rollout workers beside `harness` engines (`inference.platypus_venv`) | `arc2-worker-vllm-fork` |

The trainer environment must never contain the serving stack (vLLM,
FlashInfer, platypus, duck). The learner and the engines are separate
processes with separately qualified pins: the learner's torch, Transformers,
FLA and Triton versions come from this repository's lock, while the harness lock
pins a vLLM fork with its own torch/FlashInfer stack. The trainer process does
not import vLLM (see "Process split" in `training/FLASH_RL.md`; the
`without_vllm` import probe enforces it), and `pyproject.toml` declares the
`harness` and `flash-distillation` extras as conflicting. Installing both
stacks in one venv would let a serving upgrade silently change the learner.
The trainer recipe therefore exports only the requested extras, installs the
harness's `ullm` package with `--no-deps` (the multimodal path imports
`ullm.media`, pure Python over PIL/numpy/torch/Transformers), and fails if any
`--forbid` module (default `vllm flashinfer platypus duck`) is importable.
The harness environment carries no repository code: jobs put their frozen
source on `PYTHONPATH`.

ARC-2 rollout workers cannot run from the harness environment: the ARC-2
harness (arc2-platypus) defines packages named `duck`, `platypus` and `taaf`,
as arc3-platypus does, so it gets an interpreter of its own. The worker talks
to a harness engine through ullm's `EngineProxy`, which loads the tokenizer and
image processor locally and refuses to render unless they match the engine's
(`local image processing differs from engine metadata`: the processing identity
names the Transformers, torch, torchvision, Pillow and NumPy versions). So the
worker project, `environments/arc2-worker-vllm-fork`, locks the same serving
stack as the harness lock (the vLLM fork, Transformers 5.16.1, torch 2.13.0,
torchvision 0.28.0), and the build refuses a project lock that disagrees with
the harness lock on any shared distribution. Its ullm is the `platypus`
submodule's, the one the trainer and the engines use. `environments/arc2-worker`,
on the repository lock's vLLM 0.25.1 stack, is the 27B launcher's worker
(`scripts/platypus_debug_smoke.sbatch` builds it itself); it does not work
against harness engines.

## Building

> **In this monorepo.** The builders read `scripts/build_env.sbatch`, `uv.lock` and the
> `platypus` and `arc2-platypus` gitlinks at the root of a revision, which was the layout of
> our training repository. Here `training/` is a subdirectory and those two are symlinks with
> uv path sources, so the builders do not run as they stand, and `scripts/flash_rl_submit.sh`
> finds its default source and environments the same way. `uv sync` in `training/` creates
> the project's environment; this section records how the job environments were built.

Builds run on a CPU node (`--qos=dev`, 16 CPUs, 32G) from a commit of a clone
they only read. From a checkout, one command submits all three for its HEAD:

```bash
scripts/build_envs.sh [--revision REV] [--out DIR] [--repo DIR] [--force | --check] [-- SBATCH_OPTION...]
```

It plans every recipe first (a bad revision or `--out` submits nothing), creates
the artifact root (`--out`, else `$ARC_TRAINING_ENVS`, else
`/data/personal/$USER/arc-training-envs`), skips each recipe whose artifact
already matches the revision (below; `--force` builds anyway), submits the rest
with the checkout as `--repo` and the revision's own `build_env.sbatch`, and
prints the job ids, a `sacct` line to follow them and where their
`BUILD_ENV_RESULT` lines will be (`<out>/slurm-<jobid>.out`). The revision need
not be pushed, since the jobs archive it from the checkout when they start, but
the submodule commits it pins must be on their remotes, and others can only
rebuild from the recorded provenance once it is pushed. Its last lines are the
`scripts/flash_rl_submit.sh` command to run next; for a revision other than
the checkout's HEAD, or another `--repo`, that command ends with
`-- <repo> <full commit>`.

`--check` answers "is REV built?" without submitting or creating anything: it
prints, per recipe, the matching artifact or that there is none (with the
resolver's note on the nearest artifact, below), and exits 3 if any recipe has
none.

Building one recipe by hand, from a fetch-only clone; one-time setup on the
login node:

```bash
git clone --no-checkout <training-repo-url> /data/personal/$USER/arc-training-src
mkdir -p /data/personal/$USER/arc-training-envs   # artifacts and Slurm logs
```

Each build (push first; fetching is the only thing done to the clone):

```bash
SRC=/data/personal/$USER/arc-training-src
git -C "$SRC" fetch origin
REV=$(git -C "$SRC" rev-parse origin/<branch>)   # or any pushed 40-hex commit
sbatch <(git -C "$SRC" show "$REV:scripts/build_env.sbatch") trainer "$REV"
sbatch <(git -C "$SRC" show "$REV:scripts/build_env.sbatch") harness "$REV"
sbatch <(git -C "$SRC" show "$REV:scripts/build_env.sbatch") arc2-worker "$REV"
```

`sbatch scripts/build_env.sbatch trainer "$REV"` from a checkout works the
same; submitting the revision's own copy needs no checkout and keeps the
builder identical to the revision (the manifest records whether it was). The
job archives `$REV` to its scratch directory and runs the helpers
(`pack_venv.sh`, `stage_venv.sh`, `env_checks.py`) from that revision. The
Slurm log is `/data/personal/$USER/arc-training-envs/slurm-<jobid>.out`; its
last line is `BUILD_ENV_RESULT recipe=... name=... key=<key> tarball=<path>
bytes=... staged=yes`. Add `--exclude=<node>` or `--account=` to the `sbatch`
line as usual.

`bash scripts/build_env.sbatch <recipe> "$REV" [options] --print-plan` (or
`bash <(git -C "$SRC" show "$REV:scripts/build_env.sbatch") ... --print-plan`)
prints the resolved recipe (revision, lock hashes, harness revision and source,
artifact name, build and check steps, existing artifacts) without building; it
needs no Slurm and only reads the clone, so it is fine on the login node.

Options (`--help` lists them):

- trainer: `--extra NAME` (repeatable; replaces the default),
  `--harness-package NAME` (repeatable; installed `--no-deps` from
  `packages/NAME` of the harness at the submodule pin), `--no-harness-packages`,
  `--forbid MODULE` (repeatable; replaces the default list).
- harness: `--harness-revision SHA` (overrides the submodule pin, for
  experiments), `--package NAME` (default `duck`; a non-default package names the
  environment `harness-<package>`).
- arc2-worker: `--arc2-harness-clone DIR` (an existing clone holding the
  `arc2-platypus` pin; by default the job clones its submodule URL into its
  scratch). Both submodules are always at the revision's pins.
- all: `--out DIR` (default `/data/personal/$USER/arc-training-envs`),
  `--repo DIR` (default `/data/personal/$USER/arc-training-src`),
  `--harness-clone DIR` (an existing clone holding the harness revision; by
  default the job clones the submodule URL into its scratch), `--python PATH`
  (default: the shared CPython 3.12.12 seed), `--no-stage`.

GitHub access uses your own credentials on the compute node. Git sources in a
lock (the vLLM fork, the ARC framework and games) are read over the transport
the lock names, or over the other one (HTTPS credentials or SSH key) if only
that one works; the job fails early, naming the repository, if neither does.

## What a build does and checks

1. Archives the revision; clones or reads the harness revision; copies the base
   interpreter to scratch.
2. trainer: `uv export --frozen --no-dev --extra ... --no-emit-project
   --no-emit-workspace`, rejects forbidden packages in the export, installs it
   into `uv venv --relocatable` with `uv pip install --compile-bytecode`, then
   each harness package with `--no-deps`. harness: `uv sync --locked --no-dev
   --no-editable --compile-bytecode --package duck` in the harness source.
   arc2-worker: lays out `environments/arc2-worker-vllm-fork` beside both
   submodules at their pins, checks that its lock agrees with the harness
   lock, and runs `uv sync --locked --no-dev --no-editable --compile-bytecode`
   in it.
3. Checks the build tree (below), writes the provenance, packs with
   `scripts/pack_venv.sh`, publishes to `--out`.
4. Stages the artifact on the build node with `stage_venv` and repeats the
   checks from the staged copy (skipped with `--no-stage`). A build that
   publishes stages into a root of its own,
   `/cache/users/$UID/build-env-stage-<jobid>` (removed when the job ends), so
   the checks run on the tarball just written even when the node already has
   that key staged. A build whose key already exists stages as jobs do, usually
   warm, after checking the published tarball's size and sha256 against its
   manifest.

Checks (`scripts/env_checks.py`, run with the environment's own interpreter,
isolated from the source tree, CUDA hidden):

- trainer: the installed distributions equal the lock export (same versions,
  nothing missing, nothing extra except the harness packages); no `--forbid`
  module is importable and no distribution named after one is installed; each
  harness package imports, and `ullm.media` when `ullm` is installed; every
  probe in `tests/distillation/import_probes` runs from the environment with
  the revision's source on `PYTHONPATH`.
- harness: the installed distributions equal the harness lock export,
  including git sources at their locked commits and the workspace members
  (`duck`, `platypus`, `ullm`) as non-editable installs, which must import;
  `vllm.__version__` names the fork commit the harness lock pins; with the
  revision's source on `PYTHONPATH`, `training.server_endpoints`,
  `harness_support.arc3.worker` and `scripts/platypus_rollout_worker.py`
  import and the readiness CLI returns 1 for an unreachable server.
  `python -m training.native_vllm_server --help` needs vLLM to detect a GPU, so
  on the CPU build it is skipped and the log says so; the serving job exercises
  it.
- arc2-worker: before installing, every distribution the project lock shares
  with the harness lock is locked at the harness lock's version (git sources
  at its commit; `env_checks.py lock-agrees`). Then the installed
  distributions equal the project's lock export, including the vLLM fork at
  its locked commit and duck, platypus, taaf and ullm as non-editable installs,
  which must import (with `duck.agent`, `duck.presets`, `taaf.datasets`,
  `ullm.media` and `ullm.backends.engine_proxy`); `vllm.__version__` names the
  fork commit; with the revision's source on `PYTHONPATH`,
  `scripts/platypus_rollout_worker.py --help` runs as the trainer spawns it,
  `harness_support.arc2.worker`, `training_backend` and `serving_contract`
  import, duck has an engine preset for every weight mode the trainer may send
  (`training.harness.harness_engine`), and both ARC-2 datasets load at their
  pinned sizes with files byte-identical to the `arc2-platypus` pin's.

## Artifacts, keys and manifests

Artifacts land in `--out` as `<name>-<key>.tar.zst` and `<name>-<key>.json`.
The key is the first 16 hex digits of a sha256 over the name, the packing
recipe, the Python version, every installed distribution's `METADATA`, and,
for distributions not installed from an index (local packages such as
`ullm`/`duck`, git sources such as the vLLM fork), the digests of the files
they installed. Bytecode, console scripts and anything else that embeds the
build path are excluded. So:

- the same lock, extras and harness revision give the same key, wherever and
  whenever they are built. For arc2-worker both submodule pins enter the key
  through the files their packages install (duck, platypus and taaf with the
  ARC-2 datasets; ullm), so a pin that changes what is installed gives a new
  key and one that does not keeps it;
- a rebuild of an unchanged environment writes no artifact: the job
  reinstalls into scratch (minutes from the per-user uv cache), computes the
  key, finds `<name>-<key>.json`, checks the tarball against it (size and
  sha256), reports `unchanged: ... already exists (job, revision)` and still
  stages and checks that artifact. Nothing is overwritten; concurrent builds of
  one key publish once. A tarball that no longer matches its manifest fails the
  build (`pack_venv.sh` exits 4) and is left in place: neither a rebuild nor
  `--force` replaces a published key, so move its tarball and manifest aside
  and build again to republish it;
- any change in what is installed (a version, a distribution added or
  removed, or new code in a local or git package at an unchanged version) gives
  a new key and a new artifact beside the old ones. Old artifacts are never modified or
  deleted by the tooling.

Every successful build, publishing or not, also writes a receipt,
`<out>/receipts/<name>-<key>-<jobid>.json`, holding the name, key and its own
provenance (the manifest's `source` fields). The manifest keeps the provenance
of the build that first published a key; the receipts say which other
revisions' locks and pins install the same thing.

`scripts/resolve_envs.py --root DIR --repo CLONE --revision REV RECIPE...`
finds a revision's artifacts without building: it takes the identity
`build_env.sbatch --print-plan` derives for the revision (the recipe, name and
base interpreter; for the trainer the repository lock, extras, harness packages
and harness pin; for the harness its package and pin; for the ARC-2 worker its
project lock and both submodule pins) and prints, per recipe, the newest
artifact under the root whose manifest or receipt records that identity
(exit 3 if one has none). With `--run-dir` a run's own artifacts (the staged
interpreters in its `runtime_fingerprint.json`) come first. For a recipe with no
match it names, on stderr, the identity it needs and the nearest record under
the root (the manifest or receipt that differs in the fewest identity fields,
the newest among equals) with its key and each differing field's two values,
for example `the nearest is key a2de4055537dfe48 (arc2-worker-vllm-fork-a2de4055537dfe48.json),
which differs in project_uv_lock_sha256 (recorded 9a21..., this revision 5d62...)`.

The manifest records the tarball sha256 and sizes, the build-path references
`stage_venv` rewrites, and under `source` the provenance: `recipe`,
`repository`, `revision`, `uv_lock_sha256`, `requirements_sha256`, `extras`,
`harness_packages`, `forbidden_modules` (trainer), `package` and
`vllm_fork_revision` (harness), `project`, `project_uv_lock_sha256`,
`vllm_fork_revision`, `arc2_harness_repository`, `arc2_harness_revision`,
`arc2_harness_submodule_pin` (arc2-worker), `harness_repository`, `harness_revision`,
`harness_revision_source`, `harness_submodule_pin`, `harness_uv_lock_sha256`,
the uv version, the base interpreter, every installed distribution's version,
the commit of every git-installed distribution, the builder script's sha256
and whether it matches the revision's copy, and the Slurm job and node.

## Adding a dependency

- Trainer: add it to the extra in `pyproject.toml` (or to the base
  dependencies), run `uv lock`, commit `pyproject.toml` and `uv.lock`, push,
  and rebuild with `build_env.sbatch trainer <new revision>`. If it pulls a
  serving package, the build fails at the export check.
- Harness: bump the `platypus` submodule to a harness commit whose lock has the
  dependency, commit, push, and rebuild with `build_env.sbatch harness <new
  revision>` (and the trainer, if a `--harness-package` changed). For a quick
  experiment without a submodule bump, pass `--harness-revision <sha>`; the
  manifest records `harness_revision_source=--harness-revision`.
- ARC-2 worker: after bumping `arc2-platypus` or `platypus`, relock
  `environments/arc2-worker-vllm-fork` (`uv lock` in that directory, with both
  submodules checked out at their new pins; seed it with a copy of the harness
  lock, `platypus/uv.lock`, so uv keeps the harness's versions), commit, push,
  and rebuild with `build_env.sbatch arc2-worker <new revision>`. A stale lock
  fails `uv sync --locked`; a lock that drifts from the harness lock fails the
  agreement check.

Never install into a staged environment or an unpacked tarball; build a new
artifact.

## Using an environment in a job

`stage_venv` copies the tarball to `/cache/users/$UID/venvs/<name>-<key>`
under a per-key lock, checks its sha256 against the manifest, rewrites the
recorded build paths and exports the interpreter. A staged key is reused by
later jobs on the node; an evicted copy is restaged. In a batch script, after
`job-env.sh`:

```bash
. scripts/stage_venv.sh
stage_venv /data/personal/$USER/arc-training-envs/trainer-flash-distillation-<key>.tar.zst <key> ARC3_FLASH_PYTHON
```

The Flash RL jobs (`scripts/flash_rl.sbatch`, `scripts/flash_rollout_node.sh`)
stage the artifacts themselves from three variables: `ARC3_FLASH_VENV_TARBALL`
(trainer), `ARC3_HARNESS_VENV_TARBALL` (harness) and, for ARC-2 configs
(`inference.environment = "arc2"`) only, `ARC3_ARC2_WORKER_VENV_TARBALL`.
`scripts/flash_rl_submit.sh` fills each one left unset with
`scripts/resolve_envs.py` for the revision it submits, from `--envs DIR`,
`$ARC_TRAINING_ENVS` or `/data/personal/$USER/arc-training-envs`; a resumed run
keeps the artifacts it started with. It passes the job the revision's full
commit (a ref or an abbreviation is resolved once, at submission). When an
environment is missing it prints the `build_envs.sh` command to paste, with
that revision, root and clone and the submission's `--exclude` as
`-- --exclude=...`. Point `ARC_TRAINING_ENVS` at a shared root
to use environments someone else built; set a variable to a tarball path to
pin one explicitly:

```bash
export ARC3_HARNESS_VENV_TARBALL=/data/personal/$USER/arc-training-envs/harness-<key>.tar.zst
```

The trainer interpreter becomes `ARC3_FLASH_PYTHON`; the staged harness
directory becomes `PLATYPUS_VENV` for the servers, and the rollout worker's
environment (`inference.platypus_venv` in the launch and frozen configs) is the
harness for ARC-3 and the staged ARC-2 worker for ARC-2. The ARC-2 worker is
staged on the learner node only (the trainer spawns the worker there); before
anything starts, the job compares the two environments' Transformers,
tokenizers, torch, torchvision, Pillow, NumPy and vLLM versions and stops on
any difference. An archived revision has no `arc2-platypus` checkout, so the
job copies the two pinned ARC-2 datasets the trainer reads
(`harness_support/arc2/dataset.py`) from the worker's installed taaf into the
job's source tree; they enter the run's source snapshot as usual. Any artifact
whose tarball and manifest you can read can be staged, including one another
researcher built.
