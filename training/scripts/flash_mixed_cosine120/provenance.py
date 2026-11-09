"""Verify the installed harness against the checksummed shared source capsule."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path


def tree_digest(directory):
    """Hash package contents independently of their installation path."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            digest.update(str(path.relative_to(directory)).encode() + b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def imported_harness_identity():
    """Require that imports resolve to the package trees installed by bootstrap."""
    identity = json.loads(Path(os.environ['FLASH_MIXED_HARNESS_IDENTITY']).read_text())
    for name, expected in identity['packages'].items():
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f'harness package {name} is unavailable')
        actual = Path(next(iter(spec.submodule_search_locations))).resolve()
        if actual != Path(expected['path']).resolve() or tree_digest(actual) != expected['sha256']:
            raise ValueError(f'imported harness package {name} differs from its source capsule')
    return identity
