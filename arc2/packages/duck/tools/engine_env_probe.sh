#!/bin/bash
# The environment each vLLM engine process started with, read from
# /proc/<pid>/environ of every VLLM::* process (the engine core, the
# workers): whether a variable the run sets reached the processes that use
# it, independent of vLLM, ullm and the launcher. It waits (up to two hours)
# until four workers carry their names, then prints each process's values of
# the variables named. traffic_launch.py --machine-log starts it on Kaggle,
# into /kaggle/working/engine_env.txt; on the cluster, run it inside a job
# with srun --jobid JOB --overlap.
#
#   engine_env_probe.sh [VAR ...]
vars="${*:-CUDA_DEVICE_MAX_CONNECTIONS VLLM_PP_LAYER_PARTITION VLLM_USE_V2_MODEL_RUNNER}"
pattern="^($(echo "$vars" | tr ' ' '|'))="

workers() {
  local n=0 p
  for p in /proc/[0-9]*; do
    case "$(tr '\0' ' ' 2>/dev/null < "$p/cmdline")" in
      VLLM::Worker_*) n=$((n + 1)) ;;
    esac
  done
  echo "$n"
}

for _ in $(seq 1 480); do
  [ "$(workers)" -ge 4 ] && break
  sleep 15
done
echo "$(date -u +%FT%TZ) on $(hostname), variables: $vars"
for p in /proc/[0-9]*; do
  title=$(tr '\0' ' ' 2>/dev/null < "$p/cmdline")
  case "$title" in
    VLLM::*)
      values=$(tr '\0' '\n' 2>/dev/null < "$p/environ" | grep -E "$pattern" | sort | tr '\n' ' ')
      echo "pid ${p#/proc/} ${title%% *}: ${values:-(none of them)}"
      ;;
  esac
done
