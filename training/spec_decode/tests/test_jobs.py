"""The cluster job must execute the submitted trainer, not a harness copy."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from drafter import jobs


@pytest.mark.parametrize('module', ['probe', 'nvfp4'])
@pytest.mark.parametrize('revision,status,returncode', [
    ('revision', '', 0), ('revision', ' M drafter/nvfp4.py', 0), ('', '', 1),
])
def test_submission_freezes_source_and_worker_import(tmp_path, monkeypatch, module, revision, status, returncode):
    """A queued job survives checkout edits and ignores the old harness package."""
    source = tmp_path / 'checkout' / 'spec_decode'
    (source / 'drafter').mkdir(parents=True)
    (source / 'tools').mkdir()
    (source / 'pyproject.toml').write_text('[project]\nname = "drafter"\n')
    (source / 'uv.lock').write_text('version = 1\n')
    (source / 'drafter/__init__.py').write_text('')
    module_path = f'drafter/{module}.py'
    (source / module_path).write_text('from pathlib import Path\nprint(Path(__file__).resolve())\n')
    (source / 'tools/run_job.sh').write_text('echo worker\n')
    monkeypatch.setattr(jobs, '__file__', str(source / 'drafter/jobs.py'))
    monkeypatch.setenv('DRAFTER_ROOT', str(tmp_path / 'results'))
    monkeypatch.setenv('DRAFTER_MAX_TOKENS', '65536')
    commands = []
    real_run = subprocess.run

    def submit(command, **kwargs):
        """Replace scheduler calls; retain a real interpreter for the worker."""
        commands.append(command)
        if command[0] == 'sbatch':
            return SimpleNamespace(returncode=0, stdout='1234;cluster\n', stderr='')
        return SimpleNamespace(returncode=returncode, stdout=revision if 'rev-parse' in command else status, stderr='')

    monkeypatch.setattr(jobs.subprocess, 'run', submit)
    out = jobs.submit('test', f'drafter.{module}', [], node='dgx04', minutes=10, gpus=4, after='12:13',
                      runtime_python='/cache/worker-only/venv/bin/python')
    assert '--gres=gpu:b200:4' in commands[-1]
    assert '--nodelist=dgx04' in commands[-1]
    assert '--dependency=afterok:12:13' in commands[-1]
    assert (out / 'job_id').read_text() == '1234\n'
    manifest = json.loads((out / 'source_manifest.json').read_text())
    assert manifest['repository'] == 'arc-training'
    assert manifest['files_sha256'][module_path]
    assert manifest['revision'] == (revision if returncode == 0 else None)
    assert manifest['git_status'] == (status if returncode == 0 else None)
    assert manifest['dirty'] == (bool(status) if returncode == 0 else None)
    assert "'DRAFTER_MAX_TOKENS': '65536'" in (out / 'run_in_worker.py').read_text()
    (source / module_path).write_text('raise RuntimeError("edited after submission")\n')
    harness = tmp_path / 'harness'
    (harness / 'drafter').mkdir(parents=True)
    (harness / 'drafter/__init__.py').write_text('raise RuntimeError("old harness trainer loaded")\n')
    env = dict(os.environ, PYTHONPATH=f'{out / "source"}{os.pathsep}{harness}')
    result = real_run([sys.executable, str(out / 'run_in_worker.py')], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(out / 'source' / module_path)


def test_empty_root_uses_shared_default(monkeypatch):
    """An empty exported root has the same meaning as the shell's :- default."""
    from pathlib import Path

    monkeypatch.setenv('DRAFTER_ROOT', '')
    monkeypatch.setattr(jobs.getpass, 'getuser', lambda: 'test-user')
    paths = []

    def mkdir(path, **kwargs):
        """Stop before touching the shared filesystem."""
        paths.append(path)
        raise OSError('stop before creating output')

    monkeypatch.setattr(Path, 'mkdir', mkdir)
    with pytest.raises(OSError, match='stop before creating output'):
        jobs.submit('test', 'drafter.nvfp4', [], node=None, minutes=10, gpus=1)
    assert paths[0].parent == Path('/data/shared/research/arc_3_results/test-user/drafter/jobs')


def test_serving_steps_require_explicit_runtime(tmp_path, monkeypatch):
    """Fail before queueing when no qualified engine was selected."""
    monkeypatch.delenv('DRAFTER_SERVING_PYTHON', raising=False)
    monkeypatch.setenv('DRAFTER_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='serving steps require'):
        jobs.submit('extract', 'drafter.extract', [], node=None, minutes=10, gpus=1)
    assert not list(tmp_path.iterdir())


def test_gpu_training_does_not_import_harness(tmp_path, monkeypatch):
    """Four-rank startup remains available with no ullm/vLLM installation."""
    from drafter import train

    calls = []
    monkeypatch.setattr(train.torch.multiprocessing, 'spawn', lambda *a, **kw: calls.append((a, kw)))
    # An accidental import would fail even in a developer environment containing ullm.
    monkeypatch.setitem(sys.modules, 'ullm.backends.vllm_factory', None)
    train.main([str(tmp_path), '--train', 'cache:dataset', '--gpus', '4'])
    assert calls[0][1]['nprocs'] == 4
    assert calls[0][1]['args'][0] == 4
    assert calls[0][1]['args'][1] > 0
