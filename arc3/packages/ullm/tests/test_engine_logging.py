"""The engine's processes keep ``QUIET_ENGINE_LOGGERS`` below WARNING: the
logging config the factories hand them through ``VLLM_LOGGING_CONFIG_PATH``,
read as a freshly started process reads it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ullm.backends import config_local

# A process that logs as the engine core does: through the logger of the
# fork's module whose line went quiet, and through another vLLM logger beside
# it, which must keep printing.
_LOG_BOTH = (
    "import vllm.v1.core.context_threads as quiet\n"
    "from vllm.logger import init_logger\n"
    "quiet.logger.info('the quiet line')\n"
    "init_logger('vllm.v1.core.sched.scheduler').info('the loud line')\n"
)


def _child_stdout(env: dict[str, str]) -> str:
    result = subprocess.run([sys.executable, "-c", _LOG_BOTH], env=env, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLLM_LOGGING_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))


def test_the_engine_process_drops_the_quiet_logger_and_keeps_the_rest(clean_env: None) -> None:
    unquieted = _child_stdout(dict(os.environ))
    assert "the quiet line" in unquieted and "the loud line" in unquieted

    config_local._quiet_engine_logging()
    quieted = _child_stdout(dict(os.environ))
    assert "the quiet line" not in quieted
    assert "the loud line" in quieted


def test_the_config_is_vllms_own_with_the_quiet_loggers_added(clean_env: None) -> None:
    import vllm.logger

    config_local._quiet_engine_logging()
    config = json.loads(Path(os.environ["VLLM_LOGGING_CONFIG_PATH"]).read_text())
    expected = json.loads(json.dumps(vllm.logger.DEFAULT_LOGGING_CONFIG))
    expected["loggers"].update({name: {"level": "WARNING"} for name in config_local.QUIET_ENGINE_LOGGERS})
    assert config == expected


def test_a_config_path_the_caller_set_is_kept(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    theirs = tmp_path / "theirs.json"
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", str(theirs))
    config_local._quiet_engine_logging()
    assert os.environ["VLLM_LOGGING_CONFIG_PATH"] == str(theirs)
