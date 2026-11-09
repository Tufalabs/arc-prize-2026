"""Writing the package's safetensors files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch


def save_file(
    tensors: dict[str, torch.Tensor], path: Union[str, Path], metadata: Optional[dict[str, str]] = None
) -> None:
    """safetensors' ``save_file``, then the mode the umask gives any other
    file. ``save_file`` creates its file 0600, which on a shared tree leaves
    a checkpoint's weights and a cache readable to their writer alone, beside
    configs anyone can read."""
    from safetensors.torch import save_file as _save_file

    _save_file(tensors, str(path), metadata=metadata)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(path, 0o666 & ~umask)
