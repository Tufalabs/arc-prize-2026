# shellcheck shell=bash
# Sourced only by flash_rl.sbatch for heterogeneous allocations.
allocation_gpus() {
  local info count
  info=$(timeout 15s scontrol show job -o "$1" | awk -v id="$1" '$1 == "JobId=" id {print}')
  count=$(printf '%s' "$info" | sed -nE 's/.*TresPerNode=gres[:/]gpu(:[[:alnum:]_-]+)?:([0-9]+).*/\2/p')
  if [[ -z "$count" && "${ARC3_DRY_RUN:-0}" == 1 ]]; then count="$2"; fi
  [[ "$count" =~ ^[1-8]$ ]] || { echo "Cannot resolve 1..8 GPUs for allocation $1" >&2; return 1; }
  printf '%s\n' "$count"
}
prepare_het() {
  local group nodes_var id_var count nodes group_nodes nodes_csv
  SERVER_NODES=() GROUP_NODE_COUNTS=() GROUP_JOB_IDS=()
  LEARNER_GPUS=$(allocation_gpus "${SLURM_JOB_ID_HET_GROUP_0:?}" "${ARC3_LEARNER_GPUS:-}")
  SERVER_GPUS_PER_NODE=""
  for (( group=1; group<SLURM_HET_SIZE; group++ )); do
    nodes_var="SLURM_JOB_NODELIST_HET_GROUP_$group"
    id_var="SLURM_JOB_ID_HET_GROUP_$group"
    nodes=$(timeout 15s scontrol show hostnames "${!nodes_var}")
    read -r -a group_nodes <<< "${nodes//$'\n'/ }"
    [[ ${#group_nodes[@]} -gt 0 ]] || { echo "Empty server group $group" >&2; return 1; }
    GROUP_NODE_COUNTS+=("${#group_nodes[@]}")
    GROUP_JOB_IDS+=("${!id_var}")
    count=$(allocation_gpus "${!id_var}" "${ARC3_SERVER_GPUS_PER_NODE:-}")
    if [[ -n "$SERVER_GPUS_PER_NODE" && "$count" != "$SERVER_GPUS_PER_NODE" ]] || \
       [[ -n "${ARC3_SERVER_GPUS_PER_NODE:-}" && "$count" != "$ARC3_SERVER_GPUS_PER_NODE" ]]; then
      echo 'Server groups must agree on GPUs/node and ARC3_SERVER_GPUS_PER_NODE' >&2; return 1
    fi
    SERVER_GPUS_PER_NODE="$count"
    SERVER_NODES+=("${group_nodes[@]}")
  done
  NUM_SERVERS=$(( ${#SERVER_NODES[@]} * SERVER_GPUS_PER_NODE ))
  export ARC3_SERVER_GPUS_PER_NODE="$SERVER_GPUS_PER_NODE"
  export ARC3_SERVER_SEGMENT_ID="$SLURM_JOB_ID"
  export OMP_NUM_THREADS=$(( CMAKE_BUILD_PARALLEL_LEVEL / LEARNER_GPUS > 0 ? CMAKE_BUILD_PARALLEL_LEVEL / LEARNER_GPUS : 1 ))
  # Clients on the learner must connect directly to cluster DNS, not a proxy.
  nodes_csv=$(IFS=,; echo "${SERVER_NODES[*]}")
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,$nodes_csv"
  export no_proxy="$NO_PROXY"
  SERVER_PORT=$((50000 + SLURM_JOB_ID % 4000))
  # Check gateways here and engines on their allocated nodes before freezing
  # any URLs. Neither probe ever reclaims another job's listener.
  local okay=0 attempt i node node_index
  for (( attempt=0; attempt<40; attempt++ )); do
    okay=1
    for (( i=0; i<NUM_SERVERS; i++ )); do
      if ! port_free $((SERVER_PORT + 100*i + 5000)); then okay=0; break; fi
    done
    if [[ "$okay" == 1 ]]; then
      node_index=0
      for (( group=1; group<SLURM_HET_SIZE; group++ )); do
        for (( node=0; node<GROUP_NODE_COUNTS[group-1]; node++ )); do
          # Use the node's system Python (stdlib only), not a learner-local venv.
          # As at server launch, keep learner-local temp paths out of the prolog.
          if ! run timeout 30s env -u TMPDIR -u TEMPDIR -u SCRATCH_DIR -u SLURM_TMPDIR \
            srun --het-group="$group" --nodes=1 --ntasks=1 --overlap \
            --nodelist="${SERVER_NODES[$node_index]}" --chdir="$RUN_DIR" \
            /usr/bin/python3 -c 'import socket, sys
base, node, gpus = map(int, sys.argv[1:])
for gpu in range(gpus):
    port = base + 100 * (node * gpus + gpu)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
        except OSError:
            sys.exit(f"engine port {port} is in use on {socket.gethostname()}")' \
            "$SERVER_PORT" "$node_index" "$SERVER_GPUS_PER_NODE"; then
            echo "Engine port probe failed on ${SERVER_NODES[$node_index]} (allocation ${GROUP_JOB_IDS[$((group-1))]}) for base $SERVER_PORT" >&2
            okay=0; break 2
          fi
          node_index=$((node_index + 1))
        done
      done
    fi
    [[ "$okay" != 1 ]] || break
    SERVER_PORT=$((SERVER_PORT + 101))
  done
  [[ "$okay" == 1 ]] || { echo 'No free learner gateway and server engine port block' >&2; return 1; }
  run timeout 60s "$ARC3_FLASH_PYTHON" -m training.job_shape "$SRC_CONFIG" --flash-job \
    --server-nodes "$nodes_csv" --server-gpus-per-node "$SERVER_GPUS_PER_NODE" \
    --learner-gpus "$LEARNER_GPUS" --rewrite-learner-devices \
    --launch-config "$LAUNCH_CONFIG" --port "$SERVER_PORT" --venv "$WORKER_VENV" > "$WORK/shape"
  # Only the environment carries the secret, including srun's --export=ALL.
  export SERVER_API_KEY="$("$ARC3_FLASH_PYTHON" -c 'import secrets; print(secrets.token_urlsafe(32))')"
  # Publications reach the servers' GPUs over NCCL from the learner's rank 0,
  # whose rendezvous is this node's name and a free port (servers connect to
  # it, as the learner's gateways connect to theirs): nothing goes by path.
}

launch_het_servers() {
  local group index offset=0 script pid
  script="$RUN_DIR/servers/$SLURM_JOB_ID/flash_rollout_node.sh"
  mkdir -p "$(dirname "$script")"
  run timeout 30s cp "$ARC3_FROZEN_REPO_ROOT/scripts/flash_rollout_node.sh" "$script"
  export SERVER_PORT SERVER_MAX_MODEL_LEN="${SERVER_CONTEXT_WINDOWS[0]}"
  for (( group=1; group<SLURM_HET_SIZE; group++ )); do
    index=$((group-1))
    local resources=()
    if [[ "${ARC3_DRY_RUN:-0}" != 1 || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      resources=(--gres="gpu:b200:$SERVER_GPUS_PER_NODE" --cpus-per-gpu=12)
    fi
    command=(srun --het-group="$group" --nodes="${GROUP_NODE_COUNTS[$index]}"
      --ntasks="${GROUP_NODE_COUNTS[$index]}" --ntasks-per-node=1 --kill-on-bad-exit=1
      "${resources[@]}" --chdir="$RUN_DIR"
      --export="ALL,ARC3_SERVER_NODE_OFFSET=$offset,ARC3_SERVER_JOB_ID=${GROUP_JOB_IDS[$index]},SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS"
      bash "$script")
    print_command "${command[@]}"
    # Do not leak learner-local temp paths to Slurm's remote task prolog.
    env -u TMPDIR -u TEMPDIR -u SCRATCH_DIR -u SLURM_TMPDIR \
      setsid "${command[@]}" > "$RUN_DIR/servers/$SLURM_JOB_ID/group_${group}.log" 2>&1 &
    SERVER_PIDS+=($!)
    offset=$((offset + GROUP_NODE_COUNTS[index]))
  done
  if [[ "${ARC3_DRY_RUN:-0}" == 1 ]]; then
    # A dry run still executes staging/preparation on every server node.
    for pid in "${SERVER_PIDS[@]}"; do wait "$pid"; done
    SERVER_PIDS=()
    for (( group=1; group<SLURM_HET_SIZE; group++ )); do cat "$RUN_DIR/servers/$SLURM_JOB_ID/group_${group}.log"; done
    for (( index=0; index<${#SERVER_NODES[@]}; index++ )); do
      [[ -f "$RUN_DIR/servers/$SLURM_JOB_ID/node_${index}.ready" ]] || return 1
    done
  fi
}

relaunch_het() {
  local group next_limit="$1" args=()
  for (( group=0; group<SLURM_HET_SIZE; group++ )); do
    if (( group > 0 )); then args+=(:); fi
    args+=(--account=research --partition=dev --qos="${ARC3_QOS:-b200-pool}"
      --time="$next_limit" --signal=B:TERM@600 --cpus-per-gpu=12 --mem-per-gpu=96G
      --output="$RUN_DIR/slurm-%j.out")
    if (( group == 0 )); then
      args+=(--nodes=1 --ntasks=1 --gres="gpu:b200:$LEARNER_GPUS")
    else
      # Kander's submit plugin requires an explicit --ntasks on GPU jobs; a
      # chained segment without it was refused (run epmean-20261011, 2026-10-12).
      args+=(--nodes="${GROUP_NODE_COUNTS[$((group-1))]}" --ntasks="${GROUP_NODE_COUNTS[$((group-1))]}"
        --ntasks-per-node=1 --gres="gpu:b200:$SERVER_GPUS_PER_NODE")
    fi
    if [[ -n "${ARC3_EXCLUDE:-}" ]]; then args+=(--exclude="$ARC3_EXCLUDE"); fi
  done
  env -u TMPDIR -u CUDA_VISIBLE_DEVICES -u ARC3_SEGMENT_DEADLINE_EPOCH \
    -u ARC3_FORCE_SNAPSHOT -u ARC3_JOB_REPO_ROOT -u SERVER_API_KEY \
    timeout 30s sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" --chdir="$RUN_DIR" \
    "${args[@]}" "$REPO_ROOT/scripts/flash_rl.sbatch" "$SOURCE_REPO" "${REVISION:-HEAD}"
}
