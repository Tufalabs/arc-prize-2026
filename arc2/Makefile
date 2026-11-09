VENV ?= .venv
PYTHON_VERSION ?= 3.12.12
# UV_PROJECT_ENVIRONMENT makes `uv sync`/`uv run` honor a non-default VENV
# (uv otherwise hardcodes ./.venv as the project environment).
UV ?= UV_PROJECT_ENVIRONMENT=$(VENV) uv
PY := $(VENV)/bin/python

SRC = packages/
TESTS = tests/
TOOLS = tools/
ALL_CODE = ${SRC} ${TESTS} ${TOOLS}

TEST_ARGS ?=
# Hash randomisation off for every tier: recorded expectations replay text
# produced by an earlier process, and a set iterated anywhere on the way to
# one orders by hash. The seed is read at interpreter startup, so it has to
# be in the environment rather than in a fixture.
PYTEST = PYTHONHASHSEED=0 $(PY) -m pytest
# Cheap-tier parallelism, sized against the machine and capped where the
# return flattens on a suite dominated by process spawns. Deliberately NOT in
# pytest's addopts — the gpu tier must stay serial, one engine per allocated
# GPU. Override with `make test CHEAP_PARALLEL=-n0` to debug serially.
CHEAP_WORKERS ?= $(shell python3 -c "import os;print(min(12,len(os.sched_getaffinity(0))))" 2>/dev/null || echo 4)
CHEAP_PARALLEL ?= -n $(CHEAP_WORKERS)

$(PY):
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)

install: $(PY)
	$(UV) sync --python $(PY) --locked --no-dev --all-packages

install-dev: $(PY)
	$(UV) sync --python $(PY) --locked --all-packages
	$(UV) run nbstripout --install
	$(UV) run python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (arc2-platypus)"

test: install-dev
	$(PYTEST) $(CHEAP_PARALLEL) $(TEST_ARGS)

# --- datasets -------------------------------------------------------------
# The packed .npz files are committed (a fifth of a megabyte); this rebuilds
# them from upstream. The checkout is scratch, the artifacts are the product.
ARC_AGI_2_REPO ?= https://github.com/arcprize/ARC-AGI-2.git
DATA_CHECKOUT ?= .data-src/ARC-AGI-2

fetch-data: install-dev
	@if [ -d $(DATA_CHECKOUT)/.git ]; then git -C $(DATA_CHECKOUT) pull --ff-only; \
	else mkdir -p $(dir $(DATA_CHECKOUT)); git clone --depth 1 $(ARC_AGI_2_REPO) $(DATA_CHECKOUT); fi
	$(PY) tools/build_datasets.py --source $(DATA_CHECKOUT)

# --- expensive test tiers -------------------------------------------------
# Each is opt-in and gated twice: a pytest mark selects it, and the tier's
# own runtime guard fails loudly where the resource is missing. `test-full`
# runs every tier and is the pre-merge gate.
#
# pytest exits 5 when a selection matches nothing, and every tier below is
# empty until its first marked test lands. The targets tolerate exactly that
# code and say so; the tolerance goes away as each tier fills (roadmap.md).
TIER_EMPTY_OK = rc=$$?; if [ $$rc -eq 5 ]; then echo "  (tier still empty — no test carries this mark yet)"; else exit $$rc; fi

# The gpu tier runs where it is invoked: inside an interactive allocation, or
# on a node the caller already holds. From a login node use `test-gpu-srun`,
# which stages the repo to shared storage and re-enters under srun — the
# checkout itself lives on login-node-local /scratch, which a compute node
# cannot see.
test-gpu-only:
	@$(PYTEST) -m gpu -v --durations=0 --durations-min=0.05 $(TEST_ARGS); $(TIER_EMPTY_OK)

test-gpu: install-dev test-gpu-only

# Dispatch the tier to compute nodes, in two shapes. The seam checks run small
# models and fit any card; the duck's production engine is 27B at FP8 with
# the run's concurrent game hosts beside it, so it needs a B200 and host RAM
# to match. One shape for both would either starve the duck or send every
# two-minute column to a DGX node.
#
# Sequential in `test-full`, as upstream runs them: they are the two slowest
# things in the gate and running them at once would be tempting, but a shared
# queue makes the wall-clock saving unreliable and the interleaved logs are
# markedly harder to read when a column fails.
#
# Both take the same TEST_ARGS as the rest; the srun shape is overridable
# through ARC2_GPU_* (see tools/test_gpu_node.sh).
test-gpu-srun:
	@ARC2_GPU_LABEL=small ARC2_GPU_MARK="gpu and not b200" tools/test_gpu_node.sh $(TEST_ARGS)

# The duck-engine column boots Flash-Next on the B200 and plays the easy set
# on it three times: sized as the demo notebook sizes that column (its tier and
# pinned PLE table sit unreclaimable in the job's cgroup), with time for the
# three runs after one boot.
test-duck-engine-srun:
	@ARC2_GPU_LABEL=b200 ARC2_GPU_MARK="gpu and b200" \
	 ARC2_GPU_TYPE=b200 ARC2_GPU_CPUS=24 ARC2_GPU_MEM=260G ARC2_GPU_TIME=01:30:00 \
	 tools/test_gpu_node.sh $(TEST_ARGS)

test-api: install-dev
	@ULLM_API_TESTS=1 $(PYTEST) -m api $(TEST_ARGS); $(TIER_EMPTY_OK)

# Tests that need a machine we control: they start real work in subprocesses
# and assert on what the scheduler does with it, which a shared runner cannot
# be held to. Serial deliberately — the point is the contention between the
# processes under test, not between them and xdist.
test-local: install-dev
	@$(PYTEST) -m local $(TEST_ARGS); $(TIER_EMPTY_OK)

# prepare-ci (not bare test) so the pre-merge gate also runs the side checks
# — check-format, lint, pyright. A test-only gate lets a static-red merge
# through.
test-full: prepare-ci test-api test-local test-gpu-srun test-duck-engine-srun

# The diagnostics viewer over a results tree (the demo notebook's SLURM
# root by default): serves `platypus.viewer`, rendering in-flight runs from
# their span streams on every request. Falls back to a free port if 8765
# is taken. Override the root with `make viewer VIEWER_ROOT=...`.
VIEWER_ROOT ?= /data/shared/research/arc_2_results
VIEWER_PORT ?= 8765

viewer: install-dev
	$(PY) -m platypus.viewer $(VIEWER_ROOT) --port $(VIEWER_PORT)

format: install-dev
	$(UV) run ruff check ${ALL_CODE} --select I --fix
	$(UV) run ruff format ${ALL_CODE}

check-format: install-dev
	$(UV) run ruff check ${ALL_CODE} --select I
	$(UV) run ruff format --check ${ALL_CODE}

lint: install-dev
	$(UV) run ruff check ${ALL_CODE}

# grep the whole output, not tail -n1: pyright prints an upgrade nag after
# its summary line whenever a newer release exists on npm, which would
# otherwise become the last line and fail the check on every new pyright.
static: install-dev
	@output=$$( $(UV) run pyright ${ALL_CODE} 2>&1 ) ; rc=$$? ; \
		printf '%s\n' "$$output" ; \
		if [ $$rc -ne 0 ]; then exit $$rc ; fi ; \
		if ! printf '%s\n' "$$output" | grep -qE '^0 errors, 0 warnings, 0 informations\b'; then \
			echo "ERROR: pyright reported warnings; zero errors AND warnings required" >&2 ; \
			exit 1 ; \
		fi

prepare: format lint static test

prepare-ci: check-format lint static test

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache

.PHONY: install install-dev test fetch-data test-gpu-only test-gpu test-gpu-srun test-duck-engine-srun test-api test-local test-full viewer format check-format lint static prepare prepare-ci clean
