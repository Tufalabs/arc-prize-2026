.DEFAULT_GOAL := train

.PHONY: sync-harness dev-install train sft-check sft-train test lint tensorboard stop-tensorboard ensure_runs_link clean status follow cluster_run cluster_view_queue

UV ?= uv
UV_RUN ?= env -u VIRTUAL_ENV $(UV) run --locked --no-sync
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DEV_VENV_ROOT ?= $(REPO_ROOT)/.venvs
ROOT_DEV_VENV ?= $(DEV_VENV_ROOT)/root
SFT_VENV ?= $(DEV_VENV_ROOT)/sft
ROOT_DEV_UV_RUN = env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$(ROOT_DEV_VENV)" $(UV) run --locked --no-sync
SFT_UV = env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$(SFT_VENV)" $(UV)
CLUSTER_USER ?= $(if $(strip $(USER)),$(USER),$(shell whoami))
SHARED_STORAGE_ROOT ?= /data/personal/$(CLUSTER_USER)
LOCAL_SCRATCH_DIR ?= $(if $(strip $(SLURM_JOB_ID)),/scratch/jobs/$(SLURM_JOB_ID),/scratch/$(CLUSTER_USER))
CONFIG_PATH ?= configs/ppo_5gpu_arc3_officials_platypus.json
TENSORBOARD_PORT ?= 6036
TENSORBOARD_HOST ?= 0.0.0.0
RUNS_DIR := $(REPO_ROOT)/runs
TENSORBOARD_LOGDIR ?= $(RUNS_DIR)
TENSORBOARD_LOGDIR_SPEC ?=
TENSORBOARD_SCRATCH_ROOT ?= $(LOCAL_SCRATCH_DIR)/arc3-tensorboard
TENSORBOARD_INDEX_DIR ?= $(TENSORBOARD_SCRATCH_ROOT)/index
TENSORBOARD_INDEX_REFRESH_SECONDS ?= 60
TENSORBOARD_RELOAD_INTERVAL ?= 60
RUNS_TARGET ?= $(SHARED_STORAGE_ROOT)/training_runs

HF_HOME ?= $(if $(wildcard $(SHARED_STORAGE_ROOT)),$(SHARED_STORAGE_ROOT),$(HOME)/.cache/huggingface)
HF_HUB_CACHE ?= $(HF_HOME)/hub
TRANSFORMERS_CACHE ?= $(HF_HUB_CACHE)
HF_DATASETS_CACHE ?= $(HF_HOME)/datasets
HF_ASSETS_CACHE ?= $(HF_HOME)/assets
HF_XET_CACHE ?= $(HF_HOME)/xet
HUGGINGFACE_HUB_CACHE ?= $(HF_HUB_CACHE)
HUGGINGFACE_ASSETS_CACHE ?= $(HF_ASSETS_CACHE)

SLURM_NTASKS ?= 1
SLURM_GPU_TYPE ?= b200
SLURM_CPUS_PER_GPU ?= 8
SLURM_MEM_PER_GPU ?= 96G
SLURM_ACCOUNT ?= research

SLURM_RUN_PARTITION ?= interactive
SLURM_RUN_ACCOUNT ?= $(SLURM_ACCOUNT)
SLURM_RUN_QOS ?= interactive
SLURM_RUN_TIME ?= 7:00:00
SLURM_RUN_GPUS ?= 1
SLURM_RUN_GRES = $(if $(strip $(SLURM_GPU_TYPE)),gpu:$(SLURM_GPU_TYPE):$(SLURM_RUN_GPUS),gpu:$(SLURM_RUN_GPUS))
SLURM_RUN_PARTITION_ARG = $(if $(strip $(SLURM_RUN_PARTITION)),--partition=$(SLURM_RUN_PARTITION),)
SLURM_RUN_ACCOUNT_ARG = $(if $(strip $(SLURM_RUN_ACCOUNT)),--account=$(SLURM_RUN_ACCOUNT),)
SLURM_RUN_QOS_ARG = $(if $(strip $(SLURM_RUN_QOS)),--qos=$(SLURM_RUN_QOS),)

export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export HF_DATASETS_CACHE
export HF_ASSETS_CACHE
export HF_XET_CACHE
export HUGGINGFACE_HUB_CACHE
export HUGGINGFACE_ASSETS_CACHE

sync-harness:
	bash "$(REPO_ROOT)/scripts/sync_platypus.sh"

dev-install: sync-harness
	mkdir -p "$(DEV_VENV_ROOT)"
	$(UV) venv --allow-existing --python 3.12 "$(ROOT_DEV_VENV)"
	env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$(ROOT_DEV_VENV)" $(UV) sync --locked --extra dev --extra sft

ensure_runs_link:
	mkdir -p "$(RUNS_TARGET)"
	@if [ -L "$(RUNS_DIR)" ]; then \
		current_target="$$(readlink "$(RUNS_DIR)")"; \
		if [ "$$current_target" != "$(RUNS_TARGET)" ]; then \
			rm "$(RUNS_DIR)"; \
			ln -s "$(RUNS_TARGET)" "$(RUNS_DIR)"; \
		fi; \
	elif [ -e "$(RUNS_DIR)" ]; then \
		echo "Refusing to replace existing non-symlink $(RUNS_DIR)"; \
		exit 1; \
	else \
		ln -s "$(RUNS_TARGET)" "$(RUNS_DIR)"; \
	fi

train:
	$(UV_RUN) python -m training.launch_frozen --config "$(CONFIG_PATH)"

# Per-action SFT from TAAF traces (sft/). sft-check builds/inspects the dataset
# with no model load (light: datasets+pillow); sft-train runs the TRL trainer
# (heavy: + trl/accelerate, GPU). SFT_CHECK_ARGS can add e.g. --verify-tokens.
# SFT_EXTRA_ARGS accepts additional uv sync flags, e.g. --extra=sft-rearc.
SFT_CONFIG ?= configs/sft.json
SFT_EXTRA_ARGS ?=
sft-check:
	mkdir -p "$(DEV_VENV_ROOT)"
	$(UV) venv --allow-existing --python 3.12 "$(SFT_VENV)"
	$(SFT_UV) sync --locked --extra sft $(SFT_EXTRA_ARGS)
	$(SFT_UV) run --locked --no-sync python -m sft.train --config "$(SFT_CONFIG)" --check-data $(SFT_CHECK_ARGS)

sft-train:
	mkdir -p "$(DEV_VENV_ROOT)"
	$(UV) venv --allow-existing --python 3.12 "$(SFT_VENV)"
	$(SFT_UV) sync --locked --extra sft --extra sft-train $(SFT_EXTRA_ARGS)
	$(SFT_UV) run --locked --no-sync python -m sft.train --config "$(SFT_CONFIG)"

tensorboard: stop-tensorboard ensure_runs_link
	@if [ -n "$(TENSORBOARD_LOGDIR_SPEC)" ]; then \
		$(UV_RUN) tensorboard --logdir_spec "$(TENSORBOARD_LOGDIR_SPEC)" --reload_interval "$(TENSORBOARD_RELOAD_INTERVAL)" --port "$(TENSORBOARD_PORT)" --host "$(TENSORBOARD_HOST)"; \
	else \
		index_dir="$(TENSORBOARD_INDEX_DIR)"; \
		mkdir -p "$$index_dir"; \
		sync_tensorboard_index() { \
			seen_file="$$(mktemp)"; \
			find -L "$(TENSORBOARD_LOGDIR)" -mindepth 2 -maxdepth 2 -type d -name tensorboard -print 2>/dev/null | sort | while IFS= read -r tb_dir; do \
				run_name="$$(basename "$$(dirname "$$tb_dir")")"; \
				case "$$run_name" in .*) continue ;; esac; \
				printf '%s\n' "$$run_name" >> "$$seen_file"; \
				link="$$index_dir/$$run_name"; \
				if [ -L "$$link" ] || [ ! -e "$$link" ]; then \
					ln -sfn "$$tb_dir" "$$link"; \
				else \
					echo "Skipping non-symlink TensorBoard index entry $$link"; \
				fi; \
			done; \
			for link in "$$index_dir"/*; do \
				[ -e "$$link" ] || [ -L "$$link" ] || continue; \
				name="$$(basename "$$link")"; \
				if [ -L "$$link" ] && ! grep -Fxq "$$name" "$$seen_file"; then rm "$$link"; fi; \
			done; \
			rm -f "$$seen_file"; \
		}; \
		sync_tensorboard_index; \
		if [ "$(TENSORBOARD_INDEX_REFRESH_SECONDS)" -gt 0 ]; then \
			(while true; do sleep "$(TENSORBOARD_INDEX_REFRESH_SECONDS)"; sync_tensorboard_index; done) & \
			sync_pid="$$!"; \
			trap 'kill "$$sync_pid" 2>/dev/null || true' EXIT INT TERM; \
		fi; \
		echo "TensorBoard logdir: $$index_dir"; \
		$(UV_RUN) tensorboard --logdir "$$index_dir" --reload_interval "$(TENSORBOARD_RELOAD_INTERVAL)" --port "$(TENSORBOARD_PORT)" --host "$(TENSORBOARD_HOST)"; \
	fi

stop-tensorboard:
	@$(REPO_ROOT)/scripts/stop_tensorboard.sh --port "$(TENSORBOARD_PORT)"

status:
	@latest_run="$$(find -L "$(RUNS_DIR)" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"; \
	echo "Host: $$(hostname)"; \
	echo "Note: process checks are local to this host."; \
	echo "Latest run: $${latest_run:-<none>}"; \
	if [ -n "$$latest_run" ]; then \
		echo "Training log: $$latest_run/logs.log"; \
		tail -n 30 "$$latest_run/logs.log" 2>/dev/null || true; \
		if [ -f "$$latest_run/slurm.log" ]; then \
			echo "Slurm log: $$latest_run/slurm.log"; \
		fi; \
		echo "TensorBoard dir: $$latest_run/tensorboard"; \
		echo "Shared server log: $$latest_run/inference_server.log"; \
		echo "Collected rollout windows: $$(find "$$latest_run/rollouts" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"; \
		if [ -f "$$latest_run/training_state.json" ]; then \
			echo "Latest metrics:"; \
			python3 -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1])).get("latest_iteration_metrics", {}), indent=2))' "$$latest_run/training_state.json"; \
		fi; \
	fi; \
	echo "Processes:"; \
	ps -ef | rg 'arc3-train|training.cli|training.native_vllm_server|tensorboard' | rg -v 'rg arc3-train|make status' || echo "  no active training/server/tensorboard processes"

follow:
	@latest_run="$$(find -L "$(RUNS_DIR)" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"; \
	if [ -z "$$latest_run" ]; then \
		echo "No runs found under $(RUNS_DIR)"; \
		exit 1; \
	fi; \
	echo "Following run: $$latest_run"; \
	if [ -f "$$latest_run/inference_server.log" ]; then \
		tail -n 50 -f "$$latest_run/logs.log" "$$latest_run/inference_server.log"; \
	else \
		tail -n 50 -f "$$latest_run/logs.log"; \
	fi

clean:
	@if [ -L "$(RUNS_DIR)" ] || [ -d "$(RUNS_DIR)" ]; then \
		find -L "$(RUNS_DIR)" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; \
	fi

test: dev-install
	$(ROOT_DEV_UV_RUN) pytest

lint: dev-install
	$(ROOT_DEV_UV_RUN) ruff check training harness_support sft tests
	$(ROOT_DEV_UV_RUN) vulture training sft tests --min-confidence 80

cluster_run:
	srun --chdir=$(REPO_ROOT) $(SLURM_RUN_PARTITION_ARG) $(SLURM_RUN_ACCOUNT_ARG) $(SLURM_RUN_QOS_ARG) --ntasks=$(SLURM_NTASKS) --gres=$(SLURM_RUN_GRES) --cpus-per-gpu=$(SLURM_CPUS_PER_GPU) --mem-per-gpu=$(SLURM_MEM_PER_GPU) --time=$(SLURM_RUN_TIME) --pty bash -l

cluster_view_queue:
	squeue -u "$$USER" -t R,PD
