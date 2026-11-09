"""Version preparation caches by their inputs, code, and rendering/packing options."""

import hashlib
import json
from pathlib import Path


def cache_record(stage, *, inputs, options, code_files):
    """Use logical code names so moving the checkout does not invalidate a cache."""
    code_hashes = {}
    for name, path in sorted(code_files.items()):
        code_hashes[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record = {
        "schema": "flash-mixed-preparation-cache-v1",
        "stage": stage,
        "inputs": inputs,
        "options": options,
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**record, "sha256": hashlib.sha256(encoded).hexdigest()}


def cache_matches(path, expected):
    """Incomplete, legacy, corrupt, or differently versioned entries require rebuilding."""
    try:
        return json.loads(Path(path).read_text()) == expected
    except (FileNotFoundError, json.JSONDecodeError):
        return False
