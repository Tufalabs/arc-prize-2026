"""Exercise worker environment reuse and cleanup without installing CUDA."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from drafter import jobs


@pytest.fixture
def worker(tmp_path):
    """Replace uv provisioning; execute the actual launcher and worker Python."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
with open(os.environ['UV_TEST_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'pip':
    sys.exit(int(os.environ.get('UV_TEST_INSTALL_EXIT', '0')))
venv = Path(sys.argv[-1] if sys.argv[1] == 'venv' else os.environ['UV_PROJECT_ENVIRONMENT'])
(venv / 'bin').mkdir(parents=True, exist_ok=True)
(venv / 'bin/python').symlink_to(sys.executable)
(venv / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages').mkdir(parents=True)
''')
    uv.chmod(0o755)
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
               DRAFTER_CACHE_ROOT=str(tmp_path / 'cache'), UV_TEST_LOG=str(tmp_path / 'uv.log'))

    def run(name, *, runtime='', exit_code=0, install_exit=0):
        """Each submitted source has distinct code but identical dependencies."""
        job_dir = tmp_path / name
        source = job_dir / 'source'
        (source / 'drafter').mkdir(parents=True)
        (source / 'drafter/__init__.py').write_text('')
        (source / 'drafter/probe.py').write_text(
            f'from pathlib import Path\nprint({name!r}, Path(__file__).resolve())\nraise SystemExit({exit_code})\n'
        )
        (source / 'pyproject.toml').write_text('[project]\nname = "drafter"\n')
        (source / 'uv.lock').write_text('version = 1\n')
        shutil.copytree(Path(jobs.__file__).resolve().parents[1] / 'tools', source / 'tools')
        (job_dir / 'run_in_worker.py').write_text(jobs._WORKER.format(env={}, argv=['drafter.probe'], module='drafter.probe'))
        result = subprocess.run(['bash', str(source / 'tools/run_job.sh'), str(job_dir), runtime],
                                env=dict(env, UV_TEST_INSTALL_EXIT=str(install_exit)), capture_output=True, text=True)
        return result, source

    return run


def test_locked_dependencies_reused_with_separate_sources(worker, tmp_path):
    """A second snapshot runs its own package without syncing the shared venv."""
    for name in ('first', 'second'):
        result, source = worker(name)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f'{name} {source / "drafter/probe.py"}'
    commands = [json.loads(line) for line in (tmp_path / 'uv.log').read_text().splitlines()]
    assert len(commands) == 1
    assert commands[0][0] == 'sync'
    assert '--locked' in commands[0]
    assert '--no-install-project' in commands[0]
    assert len(list((tmp_path / 'cache/venvs').glob('locked-*/.ready'))) == 1


@pytest.mark.parametrize('exit_code,install_exit', [(0, 0), (17, 0), (0, 23)])
def test_private_serving_environment_removed(worker, tmp_path, exit_code, install_exit):
    """Cleanup runs on worker success/failure and on dependency-install failure."""
    result, _ = worker('serving', runtime=sys.executable, exit_code=exit_code, install_exit=install_exit)
    assert result.returncode == (install_exit or exit_code), result.stderr
    assert not list((tmp_path / 'cache/venvs').glob('serving-*'))
