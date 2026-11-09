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
# Hash randomisation off for every tier. The reference tests replay prompts
# recorded by an earlier process, and a set of strings iterated anywhere on
# the way to one — including inside model-written code running in a REPL,
# which no amount of sorting on our side reaches — orders by hash. The seed
# is read at interpreter startup, so it has to be in the environment rather
# than in a fixture; the reference tests assert it and fail fast, which is
# what a bare `pytest` runs into.
PYTEST = PYTHONHASHSEED=0 $(PY) -m pytest
# Cheap-tier parallelism: the suite is dominated by process spawns and their
# imports, which parallelize cleanly (per-test tmp dirs, per-worker semaphore
# managers).
#
# `--dist loadfile` pins a file's tests to one worker. That matters because
# the runtime is module-scoped SETUP, not calls: six scripted-reference
# modules replay a whole game at 12-20s each. Under the default `load`, tests
# from one module scatter across workers and each pays that setup again, so
# the suite does the same replay several times. Measured on a 192-core box,
# whole cheap tier at -n 12: ~69s with `load`, ~51s with `loadfile`, and
# loadfile at -n 8 (63s) already beats load at -n 12.
#
# Worker count: 142s at -n 4, 82s at -n 8, 68s at -n 12, 64s at -n 16, 58s at
# -n 24 (all `load`). Sharp to 12, flat after — the floor is the longest
# single module, which no worker count and no dist mode divides.
#
# Sized against the machine, not pinned at 12. The suite is not as
# spawn-bound as the numbers above suggest: a worker's REPL cells contend
# for the same cores, so on a four-core runner twelve of them serialise
# against four real cores. Measured: 51s here, still unfinished after twenty
# minutes on CI.
# `min` rather than a fixed number because the ceiling is what 12 buys, and
# the floor is what the box has.
#
# Deliberately NOT in pytest's addopts — the gpu tier must stay serial, one
# engine per allocated GPU. Override with `make test CHEAP_PARALLEL=-n0` to
# debug serially.
CHEAP_WORKERS ?= $(shell python3 -c "import os;print(min(12,len(os.sched_getaffinity(0))))" 2>/dev/null || echo 4)
CHEAP_PARALLEL ?= -n $(CHEAP_WORKERS) --dist loadfile

$(PY):
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)

install: $(PY)
	$(UV) sync --python $(PY) --locked --no-dev --all-packages

install-dev: $(PY)
	$(UV) sync --python $(PY) --locked --all-packages
	$(UV) run nbstripout --install
	$(UV) run python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (arc3-platypus)"
	-$(UV) run python -m re_arc.official_games --optional

test: install-dev
	$(PYTEST) $(CHEAP_PARALLEL) $(TEST_ARGS)

# --- expensive test tiers -------------------------------------------------
# gpu tier needs a visible GPU (the fixtures fail loudly
# otherwise). api tier spends provider money and is double-gated on the
# ULLM_API_TESTS=1 opt-in. `test-full` runs every tier in its appropriate
# location: cheap + api locally (api calls would waste a GPU allocation),
# gpu shipped to a GPU node via srun.

# Mirrors taaf.deploy_slurm's target rendering (gres names b200/b300,
# 8 cpus + 32G per gpu). Override GPU_SRUN wholesale for another tier.
#
# `dev`, not `short`: the columns are minutes of work, so the shorter queue
# reads as the right ask, but it carries lower priority here and waits
# longer for a node. Its 30-minute ceiling would bind --time as well — an
# over-long request is rejected at submit rather than trimmed.
GPU_SRUN ?= srun --ntasks=1 --account=research --gres=gpu:b200:1 --partition=dev --cpus-per-gpu=8 --mem-per-gpu=32G --time=45:00 --export=ALL


# -v + durations: per-cell lines and a per-phase timing table (all entries
# above the noise floor); session-scoped engine startup shows up as the
# setup time of each column's first test.
test-gpu-only:
	$(PYTEST) -m gpu -v --durations=0 --durations-min=0.05 $(TEST_ARGS)

# Plain locked sync (with dev deps for pytest): the GPU stage dir is an
# rsync without .git, so install-dev's `nbstripout --install` would die
# with "not a git repository"; nbstripout/ipykernel are dev-box niceties
# a test node doesn't need.
install-gpu: $(PY)
	$(UV) sync --python $(PY) --locked --all-packages

test-gpu: install-gpu test-gpu-only

# Same tier, dispatched to a GPU node. /scratch (this repo's home) is
# login-node-local, so the workspace is rsynced to shared storage first;
# tools/test_gpu_node.sh then resolves node-local storage for the venv and
# caches on the compute side (mirrors taaf.deploy_slurm).
# Keyed by this checkout's directory name: parallel worktrees run test-full
# concurrently, and a shared stage dir means one fork's rsync --delete
# clobbers the tree another fork's live GPU jobs are reading.
GPU_STAGE_ROOT ?= /data/personal/$(USER)/arc3-platypus-gpu-stage-$(notdir $(CURDIR))
GPU_STAGE_DIR ?= $(GPU_STAGE_ROOT)/arc3
# This project's path dependencies sit beside it, so they are staged beside it.
GPU_STAGE_SIBLINGS = rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
	--exclude .ruff_cache ../re-arc-3 ../tufa-arc-agi-framework $(GPU_STAGE_ROOT)/

# One srun job per backend column, in parallel (each gets its own GPU;
# wall-clock = slowest column). Column selectors are unique -k substrings
# of the matrix ids. Per-column logs land in .gpu-test-logs/.
# The gpu_traced_* columns are the P10 trace tests, gpu_traffic the traffic
# replay's end to end, gpu_solver_engine the solver's own engine, and the tc*
# columns are the token-count ones: each boots its own engine, and stacking a
# second engine after a column's session fixture wedges engine startup (or,
# beside the matrix's 9B, does not fit the card) — so every one of them gets
# its own srun/GPU (~2 min columns).
GPU_MATRIX_COLUMNS ?= direct proxy server gpu_traced_vllmx gpu_traced_oai gpu_traffic gpu_solver_engine tcengine tchttp victim_offload_engine

test-gpu-srun:
	mkdir -p $(GPU_STAGE_DIR) .gpu-test-logs
	$(GPU_STAGE_SIBLINGS)
	rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ \
		--exclude .pytest_cache --exclude .ruff_cache --exclude .gpu-test-logs --exclude demo_runs \
		./ $(GPU_STAGE_DIR)/
	@set -e; pids=""; \
	for c in $(GPU_MATRIX_COLUMNS); do \
		echo "launching column $$c"; \
		$(GPU_SRUN) $(GPU_STAGE_DIR)/tools/test_gpu_node.sh -k $$c \
			> .gpu-test-logs/$$c.log 2>&1 & pids="$$pids $$!:$$c"; \
	done; \
	rc=0; \
	for entry in $$pids; do \
		p=$${entry%%:*}; c=$${entry##*:}; \
		if wait $$p; then echo "column $$c: OK"; \
		else echo "column $$c: FAILED (see .gpu-test-logs/$$c.log)"; rc=1; fi; \
	done; \
	exit $$rc

# A production duck engine on a B200: the 27B column starts, beats taps
# level 1, and plays traced taps games whose steps nest (U5). Sized as the
# demo notebook sizes a production column: the host tier the cell claims
# eagerly at start sits unreclaimable in the step's cgroup. (No
# serving-bug check tests by policy — see EngineFactory.verify_serving_bug.)
DUCK_ENGINE_SRUN ?= srun --ntasks=1 --account=research --gres=gpu:b200:1 --partition=dev --cpus-per-gpu=24 --mem-per-gpu=260G --time=1:00:00 --export=ALL
DUCK_ENGINE_COLUMNS ?= duck_direct_boot

test-duck-engine-srun:
	mkdir -p $(GPU_STAGE_DIR) .gpu-test-logs
	$(GPU_STAGE_SIBLINGS)
	rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ \
		--exclude .pytest_cache --exclude .ruff_cache --exclude .gpu-test-logs --exclude demo_runs \
		./ $(GPU_STAGE_DIR)/
	@set -e; pids=""; \
	for c in $(DUCK_ENGINE_COLUMNS); do \
		echo "launching duck-engine column $$c"; \
		$(DUCK_ENGINE_SRUN) $(GPU_STAGE_DIR)/tools/test_gpu_node.sh -k $$c \
			> .gpu-test-logs/$$c.log 2>&1 & pids="$$pids $$!:$$c"; \
	done; \
	rc=0; \
	for entry in $$pids; do \
		p=$${entry%%:*}; c=$${entry##*:}; \
		if wait $$p; then echo "duck-engine column $$c: OK"; \
		else echo "duck-engine column $$c: FAILED (see .gpu-test-logs/$$c.log)"; rc=1; fi; \
	done; \
	exit $$rc

# The seal's production engine on a B200: it starts and beats taps level 1.
# The duck column's sizing, which is already the Flash-Next engine's.
SEAL_ENGINE_SRUN ?= $(DUCK_ENGINE_SRUN)
SEAL_ENGINE_COLUMNS ?= seal_direct_boot

test-seal-engine-srun:
	mkdir -p $(GPU_STAGE_DIR) .gpu-test-logs
	$(GPU_STAGE_SIBLINGS)
	rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ \
		--exclude .pytest_cache --exclude .ruff_cache --exclude .gpu-test-logs --exclude demo_runs \
		./ $(GPU_STAGE_DIR)/
	@set -e; pids=""; \
	for c in $(SEAL_ENGINE_COLUMNS); do \
		echo "launching seal-engine column $$c"; \
		$(SEAL_ENGINE_SRUN) $(GPU_STAGE_DIR)/tools/test_gpu_node.sh -k $$c \
			> .gpu-test-logs/$$c.log 2>&1 & pids="$$pids $$!:$$c"; \
	done; \
	rc=0; \
	for entry in $$pids; do \
		p=$${entry%%:*}; c=$${entry##*:}; \
		if wait $$p; then echo "seal-engine column $$c: OK"; \
		else echo "seal-engine column $$c: FAILED (see .gpu-test-logs/$$c.log)"; rc=1; fi; \
	done; \
	exit $$rc

test-api: install-dev
	ULLM_API_TESTS=1 $(PYTEST) -m api $(TEST_ARGS)

# Tests that need a machine we control: they start real games in
# subprocesses and assert on what the scheduler does with them, which a
# shared runner cannot be held to. Serial deliberately — the point is the
# contention between the games under test, not between them and xdist.
test-local: install-dev
	$(PYTEST) -m local $(TEST_ARGS)

# prepare-ci (not bare test) so the pre-merge gate also runs the side
# checks — check-format, lint, pyright — a test-only gate once let a
# static-red merge through.
test-full: prepare-ci test-api test-local test-gpu-srun test-duck-engine-srun test-seal-engine-srun

# The diagnostics viewer over the shared results tree (serve_diagnostics
# successor). One server, two front ends: the canonical diagnostics pages
# at / and the ARC3 reasoning replay at /_replay/ — serve() prints both
# clickable URLs (VS Code auto-forwards); falls back to a free port if
# 8765 is taken. Override the root with `make viewer VIEWER_ROOT=...`.
VIEWER_ROOT ?= /data/shared/research/arc_3_results
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

.PHONY: install install-dev test install-gpu test-gpu test-gpu-srun test-duck-engine-srun test-seal-engine-srun test-api test-local test-full viewer format check-format lint static prepare prepare-ci clean
