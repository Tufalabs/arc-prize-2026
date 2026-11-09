from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from training.config import ExperimentConfig, require_server_endpoints
from training.run_artifacts import setup_experiment_directory
from training.source_snapshot import FrozenRunPaths, prepare_frozen_run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pythonpath_with_snapshot(source_root: Path, inherited: str | None) -> str:
    paths = [str(source_root)]
    if inherited:
        paths.append(inherited)
    return os.pathsep.join(paths)


def _env_flag_enabled(name: str) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _training_python(config: ExperimentConfig, default: str) -> str:
    if config.training.weight_mode != "flash":
        return default
    interpreter = str(os.environ.get("ARC3_FLASH_PYTHON", "")).strip()
    if not interpreter or not Path(interpreter).is_absolute():
        raise ValueError("ARC3_FLASH_PYTHON must name the absolute Flash runtime interpreter")
    return interpreter


def _exec_training_from_snapshot(
    *,
    paths: FrozenRunPaths,
    run_dir: str | None,
    argv0_python: str,
) -> None:
    env = os.environ.copy()
    env["ARC3_RUN_DIR"] = str(paths.run_root)
    env.setdefault("ARC3_ORIGINAL_REPO_ROOT", str(paths.live_repo_root))
    env["ARC3_FROZEN_REPO_ROOT"] = str(paths.source_root)
    env["ARC3_FROZEN_CONFIG_PATH"] = str(paths.frozen_config_path)
    env["ARC3_RUNTIME_FINGERPRINT_PATH"] = str(paths.runtime_fingerprint_path)
    env["PYTHONPATH"] = _pythonpath_with_snapshot(
        paths.source_root,
        env.get("PYTHONPATH"),
    )
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("FLA_TILELANG", "0")
    env.setdefault("FLA_CONV_BACKEND", "triton")

    config = ExperimentConfig.load(paths.frozen_config_path)
    require_server_endpoints(config.inference)
    argv0_python = _training_python(config, argv0_python)
    if config.training.uses_torchrun:
        mode = config.training.weight_mode
        devices = config.training.torchrun_device_ids
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        command = [
            argv0_python,
            "-m",
            "training.full_weight_launch",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={len(devices)}",
            "--max-restarts=0",
            "--module",
            f"training.{mode}_entrypoint",
            "--config",
            str(paths.frozen_config_path),
        ]
    else:
        command = [
            argv0_python,
            "-m",
            "training.cli",
            "--config",
            str(paths.frozen_config_path),
        ]
    if run_dir:
        command.extend(["--run-dir", str(paths.run_root)])
    os.chdir(paths.source_root)
    env["PWD"] = str(paths.source_root)
    os.execvpe(argv0_python, command, env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the checkout into the run's source.tar, unpack it on the node, "
            "then launch training from that copy."
        )
    )
    parser.add_argument(
        "--config", required=True, help="Path to the training config JSON file."
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Use this run directory instead of creating a timestamped one. "
            "If it already has a source archive, that archive is reused."
        ),
    )
    parser.add_argument(
        "--force-snapshot",
        action="store_true",
        help="Replace any existing source snapshot before launching.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Prepare the run without starting training. Print only the absolute "
            "source directory to stdout so a job can launch its servers and "
            "trainer from the same snapshot."
        ),
    )
    args = parser.parse_args()

    live_repo_root = _repo_root()
    config = ExperimentConfig.load(args.config)
    if args.run_dir:
        os.environ["ARC3_RUN_DIR"] = args.run_dir
    run_root, _log_path, _tensorboard_dir = setup_experiment_directory(
        live_repo_root / config.training.output_dir
        if not Path(config.training.output_dir).is_absolute()
        else Path(config.training.output_dir)
    )

    force_snapshot = bool(
        args.force_snapshot or _env_flag_enabled("ARC3_FORCE_SNAPSHOT")
    )
    paths = prepare_frozen_run(
        config_path=args.config,
        live_repo_root=live_repo_root,
        run_root=run_root,
        training_python=Path(_training_python(config, sys.executable)),
        force_snapshot=force_snapshot,
    )
    log_output = sys.stderr if args.prepare_only else sys.stdout
    print(
        f"Frozen source archive: {paths.source_archive_path}",
        file=log_output,
        flush=True,
    )
    print(
        f"Frozen source unpacked at: {paths.source_root}", file=log_output, flush=True
    )
    print(f"Frozen config: {paths.frozen_config_path}", file=log_output, flush=True)
    print(
        f"Runtime fingerprint: {paths.runtime_fingerprint_path}",
        file=log_output,
        flush=True,
    )
    if args.prepare_only:
        print(paths.source_root.resolve(), flush=True)
        return
    _exec_training_from_snapshot(
        paths=paths,
        run_dir=args.run_dir,
        argv0_python=sys.executable,
    )


if __name__ == "__main__":
    main()
