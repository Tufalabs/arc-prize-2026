"""Verify shared launch capsules against the pinned source recipe before unpacking."""
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1])
    identity = json.loads((root/'training-identity.json').read_text())
    assert digest(root/'source.tar') == identity['source_sha256'], 'source capsule differs'
    expected = json.loads(Path(__file__).with_name('artifacts.json').read_text())
    for name, checksum in expected['files'].items():
        assert digest(root/name) == checksum, name


if __name__ == '__main__':
    main()
