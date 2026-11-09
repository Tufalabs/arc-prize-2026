"""Check wheel selection against the installed torch, without importing torch."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("installed_version", ["2.11.0", "2.11.0+cu130"])
def test_conv1d_wheel_uses_installed_torch_version(
    tmp_path: Path, installed_version: str
) -> None:
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" / "platypus_debug_smoke.sbatch"
    )
    assignments = [
        line.strip()
        for line in launcher.read_text().splitlines()
        if line.strip().startswith(("TORCH_VERSION=", "CONV1D_WHEEL="))
    ]
    assert len(assignments) == 2
    # The universal lock contains different torch versions for different
    # platforms. Neither textual order nor the other platform is the ABI
    # used by the installed extension.
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "torch"\nversion = "2.13.0"\n'
        '[[package]]\nname = "torch"\nversion = "2.11.0"\n'
    )
    train_venv = tmp_path / "venv"
    (train_venv / "bin").mkdir(parents=True)
    (train_venv / "bin" / "python").symlink_to(sys.executable)
    packages = tmp_path / "packages"
    metadata = packages / f"torch-{installed_version}.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: torch\nVersion: {installed_version}\n"
    )
    # If the launcher imports torch instead of querying metadata, fail:
    # selecting a wheel must not load CUDA libraries or require a GPU.
    (packages / "torch.py").write_text('raise RuntimeError("must not import torch")\n')
    cache = tmp_path / "cache"
    wheel = (
        cache
        / "wheels/py312-torch2.11.0-cu130-b200/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl"
    )
    wheel.parent.mkdir(parents=True)
    wheel.touch()
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            "\n".join(
                [
                    *assignments,
                    'test -f "$CONV1D_WHEEL"',
                    'printf "%s\\n%s\\n" "$TORCH_VERSION" "$CONV1D_WHEEL"',
                ]
            ),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "TRAIN_VENV": str(train_venv),
            "ARC3_RL_CACHE_ROOT": str(cache),
            "ARC3_CAUSAL_CONV1D_WHEEL": "",
            "PYTHONPATH": str(packages),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["2.11.0", str(wheel)]
