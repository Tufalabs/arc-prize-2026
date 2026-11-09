.DEFAULT_GOAL := run

VENV ?= .venv
PYTHON_VERSION ?= 3.12
UV ?= uv
PY := $(VENV)/bin/python

CONFIG ?= config.env
RUN_ARGS ?=
WEB_HOST ?=
WEB_PORT ?=
DSL_GIF ?= runs/dsl-$(shell date +%Y%m%d-%H%M%S).gif
DSL_MAX_ACTIONS ?= 1200
TEMP_ARGS ?=
TEST_ARGS ?=
PREPARE_TEST_ARGS ?= --ignore=tests/test_reset_restores_initial_frame.py
BASELINE_ARGS ?=
BASELINE_ACTIONS_ARGS ?=
DSL_GIFS_ARGS ?=
PIPELINE_ARGS ?=
PIPELINE_RUN_ID ?= run-$(shell date +%Y%m%d-%H%M%S)
PRECOMPUTE_DSL_ARGS ?=
PRO_LONG_ARGS ?=

REARC_SRC = re_arc/
PIPELINE_SRC = pipeline/
SCRIPTS = scripts/
TESTS = tests/
ALL_CODE = ${REARC_SRC} ${PIPELINE_SRC} ${SCRIPTS} ${TESTS}

$(PY):
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)

install: $(PY)
	$(UV) sync --python $(PY) --locked --no-dev

install-dev: $(PY)
	$(UV) sync --python $(PY) --locked --extra dev

sync-dev: $(PY)
	$(UV) sync --python $(PY) --locked --extra dev --no-install-project

run:
	$(PY) -m re_arc --config $(CONFIG) $(RUN_ARGS)

play:
	$(PY) -m re_arc --config $(CONFIG) --policy web $(if $(WEB_HOST),--host $(WEB_HOST),) $(if $(WEB_PORT),--port $(WEB_PORT),) $(RUN_ARGS)

games:
	$(PY) -m re_arc --config $(CONFIG) --list-games

dsl: dsl-gif

dsl-gif:
	mkdir -p runs
	$(PY) -m re_arc --config $(CONFIG) --policy dsl --max-actions $(DSL_MAX_ACTIONS) --gif $(DSL_GIF) $(RUN_ARGS)
	@echo "Wrote GIF to $(DSL_GIF)"

dsl-debug:
	$(PY) -m re_arc --config $(CONFIG) --policy dsl --max-actions $(DSL_MAX_ACTIONS) $(RUN_ARGS)

dsl-gifs:
	mkdir -p runs
	$(PY) -m pipeline.run_dsl_gifs --config $(CONFIG) --max-actions $(DSL_MAX_ACTIONS) $(DSL_GIFS_ARGS)

precompute_dsl_actions:
	$(PY) -m re_arc.dsl.precomputed_actions --config $(CONFIG) --sync-baseline-actions $(PRECOMPUTE_DSL_ARGS)

test: sync-dev
	$(PY) -m pytest $(TEST_ARGS)

test-prepare: sync-dev
	$(PY) -m pytest $(PREPARE_TEST_ARGS) $(TEST_ARGS)

test-reset-restores:
	$(PY) -m pytest tests/test_reset_restores_initial_frame.py $(TEST_ARGS)

baseline:
	$(PY) -m re_arc.baseline_robustness $(BASELINE_ARGS)

baseline-actions:
	$(PY) -m re_arc.dsl.precomputed_actions --config $(CONFIG) --baseline-actions-only $(BASELINE_ACTIONS_ARGS)

pipeline-specs:
	$(PY) -m pipeline.cli run --run-id $(PIPELINE_RUN_ID) $(PIPELINE_ARGS)

pipeline-implement:
	$(PY) -m pipeline.cli implement-specs --run-id $(PIPELINE_RUN_ID) $(PIPELINE_ARGS)

pipeline-run:
	$(PY) -m pipeline.cli run --run-id $(PIPELINE_RUN_ID) $(PIPELINE_ARGS)

pro-long:
	$(PY) -m re_arc.prolong_player $(PRO_LONG_ARGS)

clean:
	rm -rf runs .mypy_cache .pytest_cache .ruff_cache

.PHONY: install install-dev sync-dev run play games dsl dsl-gif dsl-debug dsl-gifs precompute_dsl_actions test test-prepare test-reset-restores baseline baseline-actions pipeline-specs pipeline-implement pipeline-run pro-long clean profile-prepare

format: sync-dev
	$(PY) -m ruff check ${ALL_CODE} --select I --fix
	$(PY) -m ruff format ${ALL_CODE}
check-format: sync-dev
	$(PY) -m ruff check ${ALL_CODE} --select I
	$(PY) -m ruff format --check ${ALL_CODE}
lint: sync-dev
	$(PY) -m ruff check ${ALL_CODE}
static: sync-dev
	$(PY) -m mypy ${REARC_SRC} ${PIPELINE_SRC}
prepare: format lint static test-prepare
prepare-ci: check-format lint static test
profile-prepare: $(PY)
	$(PY) scripts/profile_prepare.py
