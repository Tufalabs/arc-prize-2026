"""Files the package writes are as readable as any other file of the run."""

from __future__ import annotations

import os

import torch

from drafter.files import save_file
from drafter.nvfp4 import IGNORE_ADDED, quantized_ignore


def test_a_saved_file_has_the_mode_the_umask_gives(tmp_path):
    previous = os.umask(0o002)
    try:
        save_file({"a": torch.zeros(2)}, tmp_path / "group.safetensors")
        os.umask(0o077)
        save_file({"a": torch.zeros(2)}, tmp_path / "private.safetensors", metadata={"format": "pt"})
        assert os.umask(0o077) == 0o077  # left as it was found
    finally:
        os.umask(previous)
    assert (tmp_path / "group.safetensors").stat().st_mode & 0o777 == 0o664
    assert (tmp_path / "private.safetensors").stat().st_mode & 0o777 == 0o600


def test_quantizing_the_head_rewrites_an_ignore_list():
    names = ["lm_head", "mtp.*", "model.visual*", "model.mtp.*"]
    assert quantized_ignore(names, "config.json") == ["lm_head", "model.visual*", *IGNORE_ADDED]
    try:
        quantized_ignore(["lm_head", "mtp.*"], "hf_quant_config.json")
    except ValueError as error:
        assert "hf_quant_config.json" in str(error) and "model.mtp.*" in str(error)
    else:
        raise AssertionError("an ignore list without the head's patterns passed")
