"""A CUDA toolkit ullm keeps must put its nvcc where vLLM looks for it.

vLLM decides whether FlashInfer is usable by `shutil.which("nvcc")`, which
reads PATH and never CUDA_HOME. A toolkit that reaches only CUDA_HOME reads
as no FlashInfer, and every FlashInfer MoE backend is then refused as an
unsupported device.
"""

import os
import shutil
from pathlib import Path

import pytest
import ullm.backends.config_local as config_local


def _toolkit(root: Path) -> Path:
    nvcc = root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    return root


def test_inherited_cuda_home_puts_its_nvcc_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _toolkit(tmp_path / "cuda")
    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    config_local._ensure_cuda_home()

    assert shutil.which("nvcc") == str(home / "bin" / "nvcc")
    assert os.environ["CUDA_HOME"] == str(home)


def test_kept_toolkit_is_prepended_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _toolkit(tmp_path / "cuda")
    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    config_local._ensure_cuda_home()
    config_local._ensure_cuda_home()

    assert os.environ["PATH"].split(os.pathsep).count(str(home / "bin")) == 1
