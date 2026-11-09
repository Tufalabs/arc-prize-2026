"""CPU-only job allocation and shell contract, shared by frozen launchers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
from collections.abc import Mapping
from pathlib import Path

from training.config import (
    ExperimentConfig,
    evaluation_inference_config,
    max_served_sequence_tokens,
    validate_gpu_claims,
)
from training.multimodal import checkpoint_source
from training.run_artifacts import write_json_atomic


# Engine slots beyond a server's share of a window: the trainer's control-plane
# warm-up draws run beside the window's episodes.
SERVER_SEQUENCE_HEADROOM = 8


def flash_max_num_seqs(cfg: ExperimentConfig, servers: int, explicit: str | None) -> str:
    """Each Flash engine's ``--max-num-seqs``: a server's share of a window,
    ``ceil(parallel_rollouts / servers)`` at most ``inference.max_concurrent_episodes``
    (a per-server cap), plus ``SERVER_SEQUENCE_HEADROOM``. An explicit
    ``SERVER_MAX_NUM_SEQS`` replaces it but must hold the share: an engine with
    fewer slots queues the window's episodes, and collection slows without an error."""
    share = -(-cfg.parallel_rollouts // servers)
    source = f"{cfg.parallel_rollouts} attempts over {servers} servers"
    cap = cfg.inference.max_concurrent_episodes
    if cap and cap < share:
        share, source = cap, "inference.max_concurrent_episodes"
    derived = share + SERVER_SEQUENCE_HEADROOM
    if not explicit:
        return str(derived)
    if not explicit.isdigit() or int(explicit) < share:
        raise ValueError(
            f"SERVER_MAX_NUM_SEQS={explicit} must be an integer of at least {share}, a server's "
            f"share of a window ({source}); unset, the job derives {derived}"
        )
    return str(int(explicit))


def job_shape(cfg: ExperimentConfig, env: Mapping[str, str]) -> dict[str, str]:
    servers = list(cfg.inference.server_gpus or ["0"])
    rollout_servers = len(servers)
    context = max_served_sequence_tokens(cfg.inference) or 40960
    windows = [context] * len(servers)
    mode = cfg.training.weight_mode
    if cfg.official_eval.enabled:
        servers += cfg.official_eval.server_gpus
        eval_context = max_served_sequence_tokens(evaluation_inference_config(cfg)) or context
        windows += [eval_context] * len(cfg.official_eval.server_gpus)
    override = env.get("ARC3_SERVER_GPUS")
    if mode == "flash" and override:
        replacement = override.split(",")
        if len(replacement) != len(servers):
            raise ValueError("ARC3_SERVER_GPUS must preserve the Flash server count")
        servers = replacement
    elif len(servers) == 1:
        servers = [override or servers[0]]
    tp = int(env.get("ARC3_SERVER_TENSOR_PARALLEL") or "1")
    if mode == "flash" and tp != 1:
        raise ValueError("Flash rollout servers require tensor parallelism 1")
    training_gpus = set(cfg.training.device_map_gpus)
    training_gpus.update(cfg.training.torchrun_device_ids)
    for replica in cfg.training.worker_device_map_gpus:
        training_gpus.update(replica)
    if not training_gpus and str(cfg.training.device).startswith("cuda"):
        training_gpus.add(int(str(cfg.training.device).partition(":")[2] or "0"))
    separate = cfg.inference.server_allocation == "separate"
    if separate and override:
        raise ValueError("ARC3_SERVER_GPUS cannot override a separate-allocation fleet")
    seen = set(training_gpus)
    for server in servers:
        gpus = [int(gpu) for gpu in str(server).split(",")]
        if tp < 1 or len(gpus) != tp or len(set(gpus)) != len(gpus) or min(gpus) < 0:
            raise ValueError("Effective server GPU list must contain one distinct nonnegative GPU per tensor-parallel rank")
        overlap = set() if separate else seen.intersection(gpus)
        if overlap:
            raise ValueError(f"Effective server GPUs must be disjoint from trainer/eval claims; GPUs {sorted(overlap)} are claimed twice")
        if not separate:
            seen.update(gpus)
    speculative = cfg.inference.speculative_decoding.engine_config()
    shape = {
        "SERVER_MODEL": checkpoint_source(cfg.training)[0],
        "SERVER_REVISION": cfg.training.model_revision,
        "SERVER_SERVED_NAME": cfg.inference.model_id or cfg.training.model_name,
        # Flash servers run without LoRA: they apply merged policy weights.
        **({} if mode == "flash" else {"SERVER_MAX_LORA_RANK": str(cfg.training.lora.r)}),
        "SERVER_GPU_LIST": " ".join(servers),
        "SERVER_CONTEXT_LIST": " ".join(map(str, windows)),
        "DEFAULT_VISIBLE_GPUS": ",".join(str(gpu) for gpu in range(max(seen) + 1)),
        "SERVER_WEIGHT_MODE": mode,
        "SERVER_SCORE_CENTERING_TOP_K": str(cfg.training.score_centering_top_k or 0),
        "FULL_WEIGHT_TRAINER_GPUS": ",".join(map(str, cfg.training.full_weight.device_ids)) or "-",
        "TRAINER_GPUS": ",".join(map(str, sorted(training_gpus))),
        "SERVER_ENVIRONMENT": cfg.inference.environment,
        "SERVER_MULTIMODAL_CONTEXT": cfg.inference.agent_multimodal_context,
        # The fork's --speculative-config for every server; empty: no drafting.
        "SERVER_SPECULATIVE_CONFIG": json.dumps(speculative, sort_keys=True) if speculative else "",
    }
    if mode == "flash":
        shape["SERVER_MAX_NUM_SEQS"] = flash_max_num_seqs(
            cfg, rollout_servers, env.get("SERVER_MAX_NUM_SEQS"))
    if separate:
        shape["SERVER_URL_LIST"] = " ".join(cfg.inference.base_urls)
        shape["SERVER_LOG_LIST"] = " ".join(cfg.inference.server_log_paths)
    return shape


def separate_shape(cfg: ExperimentConfig, nodes: list[str], gpus_per_node: int,
                   learner_gpus: int, port: int, run_dir: Path, *,
                   rewrite_learner_devices: bool = False,
                   env: Mapping[str, str] | None = None) -> dict[str, str]:
    if cfg.training.weight_mode != "flash" or cfg.official_eval.enabled:
        raise ValueError("Separate Flash jobs require weight_mode=flash and official_eval disabled")
    if not nodes or any(not host or any(c.isspace() for c in host) for host in nodes):
        raise ValueError("Server nodes must be nonempty hostnames")
    if not 1 <= gpus_per_node <= 8 or not 1 <= learner_gpus <= 8:
        raise ValueError("Each allocation requires 1..8 GPUs per node")
    if learner_gpus != len(cfg.training.flash.device_ids) and not rewrite_learner_devices:
        raise ValueError("Learner GPU count differs from config; pass --rewrite-learner-devices")
    count = len(nodes) * gpus_per_node
    if not 1 <= port <= 65535 - 5000 - 100 * (count - 1):
        raise ValueError("Server and gateway ports must fit in 1..65535")
    cfg = copy.deepcopy(cfg)
    cfg.training.flash.device_ids = list(range(learner_gpus))
    cfg.inference.server_allocation = "separate"
    cfg.inference.server_gpus = [str(gpu) for _ in nodes for gpu in range(gpus_per_node)]
    cfg.inference.base_urls = [
        f"http://{host}:{port + 100 * (node * gpus_per_node + gpu)}"
        for node, host in enumerate(nodes) for gpu in range(gpus_per_node)
    ]
    cfg.inference.server_log_paths = [str(run_dir / f"server_{i}.log") for i in range(count)]
    validate_gpu_claims(cfg)
    # Only SERVER_MAX_NUM_SEQS applies to a separate fleet (ARC3_SERVER_GPUS is refused).
    explicit = (env or {}).get("SERVER_MAX_NUM_SEQS")
    return job_shape(cfg, {"SERVER_MAX_NUM_SEQS": explicit} if explicit else {})


def write_launch_config(source: Path, destination: Path, shape: dict[str, str], port: int, venv: str) -> None:
    payload = json.loads(source.read_text())
    inference = payload["inference"]
    inference["base_url"] = f"http://127.0.0.1:{port}"
    inference["platypus_venv"] = venv
    inference["server_gpus"] = shape["SERVER_GPU_LIST"].split()
    if "SERVER_URL_LIST" in shape:
        inference["server_allocation"] = "separate"
        inference["base_urls"] = shape["SERVER_URL_LIST"].split()
        inference["base_url"] = inference["base_urls"][0]
        payload["training"]["flash"]["device_ids"] = [int(gpu) for gpu in shape["TRAINER_GPUS"].split(",")]
    inference["server_log_paths"] = [
        str(destination.parent / f"server_{i}.log")
        for i in range(len(inference["server_gpus"]))
    ]
    write_json_atomic(destination, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--flash-job", action="store_true")
    parser.add_argument("--launch-config", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--venv")
    parser.add_argument("--server-nodes", help="Comma-separated hostnames in fleet order")
    parser.add_argument("--server-gpus-per-node", type=int)
    parser.add_argument("--learner-gpus", type=int)
    parser.add_argument("--rewrite-learner-devices", action="store_true")
    args = parser.parse_args()
    # A heterogeneous job replaces both GPU claims with its allocations' own
    # indices (separate_shape), so the source config's claims are checked
    # only after that renumbering; on one node they are checked as given.
    cfg = ExperimentConfig.load(args.config, gpu_claims=args.server_nodes is None)
    if args.server_nodes is not None:
        if None in (args.port, args.server_gpus_per_node, args.learner_gpus) or not args.launch_config:
            parser.error("--server-nodes requires GPU counts, --port and --launch-config")
        if os.environ.get("ARC3_SERVER_GPUS"):
            parser.error("ARC3_SERVER_GPUS cannot override a separate-allocation fleet")
        shape = separate_shape(cfg, args.server_nodes.split(","), args.server_gpus_per_node,
                               args.learner_gpus, args.port, args.launch_config.parent,
                               rewrite_learner_devices=args.rewrite_learner_devices,
                               env=os.environ)
    else:
        shape = job_shape(cfg, os.environ)
    if args.flash_job:
        if cfg.training.weight_mode != "flash" or cfg.official_eval.enabled:
            raise ValueError("flash_rl.sbatch requires Flash training with official_eval disabled")
        if "SERVER_URL_LIST" not in shape and (len(shape["SERVER_GPU_LIST"].split()) != 4
                or len(cfg.training.flash.device_ids) != 4
                or shape["DEFAULT_VISIBLE_GPUS"] != "0,1,2,3,4,5,6,7"):
            raise ValueError("flash_rl.sbatch requires four servers and four learner GPUs on one 8-GPU node")
    if args.launch_config:
        if args.port is None or not args.venv:
            parser.error("--launch-config requires --port and --venv")
        write_launch_config(args.config, args.launch_config, shape, args.port, args.venv)
    for name, value in shape.items():
        print(f"{name}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
