"""Exercise dependency selection and cache isolation in the RL launcher."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()


def test_trainer_sync_installs_fa4_without_distillation_stack(tmp_path: Path) -> None:
    """The actual sync command must request the kernels used by default recipes."""
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    uv.chmod(0o755)
    extras = next(
        line for line in SOURCE.splitlines() if line.startswith("TRAIN_EXTRAS=")
    )
    sync = next(
        line.strip()
        for line in SOURCE.splitlines()
        if 'UV_PROJECT_ENVIRONMENT="${TRAIN_VENV}" uv sync' in line
    )
    (tmp_path / "training").mkdir()
    capture = tmp_path / "args.json"
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", extras + "\n" + sync],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "BUILD_SRC": str(tmp_path),
            "TRAIN_VENV": str(tmp_path / "venv"),
            "CAPTURE": str(capture),
        },
        check=True,
    )
    args = json.loads(capture.read_text())
    selected = {args[i + 1] for i, value in enumerate(args) if value == "--extra"}
    assert {"harness", "fa4"} <= selected
    assert "flash-distillation" not in selected
    assert "--all-extras" not in args
    assert "--locked" in args


def test_dependency_extras_change_environment_cache_key(tmp_path: Path) -> None:
    """A cache without FA4 cannot satisfy a launch that enables FA4."""
    key = next(line for line in SOURCE.splitlines() if line.startswith("LOCK_HASH="))
    for name in (
        "uv.lock",
        "environments/arc2-worker/uv.lock",
        "environments/arc2-worker/pyproject.toml",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixed lock inputs\n")
    hashes = []
    for extras in ("--extra harness", "--extra harness --extra fa4"):
        result = subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                "\n".join(
                    [
                        'uv() { printf "uv test-version\\n"; }',
                        f"TRAIN_EXTRAS=({extras})",
                        key,
                        'printf "%s\\n" "$LOCK_HASH"',
                    ]
                ),
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "PLATYPUS_SHA": "fixed-arc3",
                "ARC2_PLATYPUS_SHA": "fixed-arc2",
                "ARC3_FAST_KERNELS": "1",
                "FAST_KERNEL_RECIPE": "fixed-kernels",
                "CUTE_CU13_RECIPE": "fixed-cute",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        hashes.append(result.stdout.strip())
    assert hashes[0] != hashes[1]
