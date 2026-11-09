import subprocess
import sys


def test_training_cli_module_invocation_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "training.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "ARC3 training pipeline" in result.stdout
    assert "--config CONFIG" in result.stdout
