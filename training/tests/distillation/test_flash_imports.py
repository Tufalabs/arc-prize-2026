"""Import boundaries: CI lacks FLA; the distillation launcher may run on Transformers 5.14.1.

The probes live in ``tests/distillation/import_probes`` as ordinary modules and
run in a child interpreter from the repository root.
"""

from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]


class FlashImportTests(unittest.TestCase):
    def run_probe(self, name):
        result = subprocess.run(
            [sys.executable, "-m", f"tests.distillation.import_probes.{name}"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_entrypoints_without_fla(self):
        self.run_probe("without_fla")

    def test_launcher_modules_without_flash_runtime(self):
        self.run_probe("launcher_modules")

    def test_rl_process_modules_without_vllm(self):
        self.run_probe("without_vllm")
